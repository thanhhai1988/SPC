
import sys
import os
import shutil
import random
import gc
import json
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_train import *
from itertools import cycle
from tqdm import tqdm
import numpy as np
from util_data import SUBSET_NAMES, TEMPLATES_SMALL
from Qwen3_VL_Embedding.src.models.qwen3_vl_embedding import Qwen3VLEmbedder

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet_100": "",
        "imagenet": "",
        "std10": "",
        "pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "eurosat": "satellite ",
        "dtd": "texture ",
        "flowers102": "flower ",
        "food101": "food ",
        "sun397": "scene ",
        "caltech101": "",
    }[dataset]
    return dataset_name

@torch.no_grad()
def get_mu_and_kappa(model, dataset):
    dataset_name = get_dataset_name_for_template(dataset)
    templates = TEMPLATES_SMALL

    all_mus =[]
    all_kappas =[]

    for class_name in SUBSET_NAMES[dataset]:
        class_texts =[]
        for template in templates:
            class_texts.append({"text": template.format(dataset_name, class_name) + "."})

        class_embs = model.process(class_texts)
        class_embs = F.normalize(class_embs, dim=-1)

        mu = class_embs.mean(dim=0)
        D = mu.shape[0]
        R = mu.norm()

        mu = mu / R
        kappa = (R * (D - R**2)) / torch.clamp(1 - R**2, min=1e-6)

        all_mus.append(mu)
        all_kappas.append(kappa)

    all_mus = torch.stack(all_mus)
    all_kappas = torch.stack(all_kappas)

    return all_mus, all_kappas

def get_image_embedding(model, images):
    image_inputs = [{"image": img} for img in images]
    image_embs = model.process(image_inputs)
    image_embs = F.normalize(image_embs, dim=-1)
    return image_embs


def get_acc(model, data_loader, mu, logit_scale, device):
    model.model.to(device)
    mu.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            labels = labels.to(device)
            image_embedding = get_image_embedding(model, images)
            # compute similarity and predict
            similarity = logit_scale * (image_embedding @ mu.T)
            preds = similarity.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def sample_tangent_gaussian(mu, kappa, num_samples, kappa_scale=0.05, kappa_max=500.0):
    C, D = mu.shape
    device = mu.device

    kappa = kappa * kappa_scale
    kappa = torch.clamp(kappa, min=1.0, max=kappa_max)

    eps = torch.randn((C, num_samples, D), device=device, dtype=mu.dtype)

    mu_expanded = mu.unsqueeze(1)
    dot_product = (eps * mu_expanded).sum(dim=-1, keepdim=True)
    eps = eps - dot_product * mu_expanded

    kappa_expanded = kappa.view(C, 1, 1)
    eps = eps / torch.sqrt(kappa_expanded)

    samples = mu_expanded + eps
    samples = F.normalize(samples, p=2, dim=-1)

    return samples

def build_text_distribution_samples(mu, kappa, num_samples=30, kappa_scale=0.5, kappa_max=5000.0):
    C, D = mu.shape
    device = mu.device
    dtype = mu.dtype
    kappa = kappa * kappa_scale
    kappa = torch.clamp(kappa, min=1.0, max=kappa_max)
    eps = torch.randn((C, num_samples, D), device=device, dtype=dtype)
    mu_expanded = mu.unsqueeze(1)
    dot_product = (eps * mu_expanded).sum(dim=-1, keepdim=True)
    eps_tangent = eps - dot_product * mu_expanded
    kappa_expanded = kappa.view(C, 1, 1)
    eps_scaled = eps_tangent / torch.sqrt(kappa_expanded + 1e-6)
    samples = mu_expanded + eps_scaled
    samples = F.normalize(samples, p=2, dim=-1)

    return samples

def logit_from_h_vectorized(logit_scale, image_feats, centroids, area_index, chosen_centroids):
    logits_base = logit_scale * (image_feats @ centroids.t())
    logits_samples = logit_scale * (image_feats @ chosen_centroids.t())
    S = chosen_centroids.shape[0]
    logits_all = logits_base.unsqueeze(0).repeat(S, 1, 1)
    logits_all[:, :, area_index] = logits_samples.t()

    return logits_all

def compute_reg_vectorized(logit_scale, area_index, samples, feats_i, centroids):
    if feats_i.shape[0] == 0:
        return torch.tensor(0.0, device=feats_i.device)

    sampled_centroids = samples[area_index]
    logits_all = logit_from_h_vectorized(logit_scale, feats_i, centroids, area_index, sampled_centroids)
    logits_flat = logits_all.reshape(-1, logits_all.size(-1))
    labels_flat = torch.full((logits_flat.size(0),), area_index, device=feats_i.device, dtype=torch.long)

    return F.cross_entropy(logits_flat, labels_flat)

def get_val_losses(
    model,
    opt_h,
    scaler,
    step,
    val_loader,
    lamda1,
    lamda2,
    lamda3,
    lambda_eps,
    use_l1,
    use_l2,
    use_l3,
    use_eps,
    writer,
    device,
    loader_iter_G=None,
    dataset="dtd",
    logit_scale=15,
    num_samples=30,
    kappa_scale=1.0
):

    model.model.eval()

    correct = 0
    total = 0

    # =========================================================
    # Dùng để tính trung bình validation loss trên toàn bộ loader
    # =========================================================
    total_val_loss = 0.0
    total_real_loss = 0.0
    total_synth_loss = 0.0
    total_eps_loss = 0.0
    total_real_reg_loss = 0.0
    total_synth_reg_loss = 0.0

    num_batches = 0

    with torch.no_grad():

        for val_images, val_labels in tqdm(val_loader):

            num_batches += 1

            val_labels = val_labels.to(device)

            # =====================================================
            # REAL EMBEDDING
            # =====================================================
            val_imgs_embedding = get_image_embedding(
                model,
                val_images
            )

            # =====================================================
            # SYNTH
            # =====================================================
            if use_l1 or use_l3 or use_eps:

                synth_images, synth_labels = next(loader_iter_G)

                synth_labels = synth_labels.to(device)

                synth_imgs_embedding = get_image_embedding(
                    model,
                    synth_images
                )

            else:

                synth_imgs_embedding = None
                synth_labels = None

            # =====================================================
            # PROTOTYPE
            # =====================================================
            mu, kappa = get_mu_and_kappa(
                model,
                dataset
            )

            # =====================================================
            # ACCURACY
            # =====================================================
            similarity = (
                logit_scale *
                (val_imgs_embedding @ mu.T)
            )

            preds = similarity.argmax(dim=1)

            correct += (
                preds == val_labels
            ).sum().item()

            total += val_labels.size(0)

            # =====================================================
            # BASE LOSS
            # =====================================================
            with torch.amp.autocast('cuda'):

                # -------------------------------------------------
                # REAL
                # -------------------------------------------------
                logits_real_all = (
                    logit_scale *
                    (val_imgs_embedding @ mu.t())
                )

                loss_real_each = F.cross_entropy(
                    logits_real_all,
                    val_labels,
                    reduction="none"
                )

                loss_real = loss_real_each.mean()

                val_loss = loss_real

                # -------------------------------------------------
                # SYNTH
                # -------------------------------------------------
                if use_l1:

                    logits_synth_all = (
                        logit_scale *
                        (synth_imgs_embedding @ mu.t())
                    )

                    loss_synth_each = F.cross_entropy(
                        logits_synth_all,
                        synth_labels,
                        reduction="none"
                    )

                    loss_synth = loss_synth_each.mean()

                    val_loss += lamda1 * loss_synth

            # =====================================================
            # EPS
            #
            # eps_i =
            #
            # 1 / (g_i * n_i)
            # sum |l(g) - l(s)|
            #
            # Chỉ tính trên các class xuất hiện ở cả validation
            # và synthetic batch
            # =====================================================
            if use_eps:

                eps_class_losses = []

                common_classes = torch.tensor(
                    list(
                        set(val_labels.tolist())
                        &
                        set(synth_labels.tolist())
                    ),
                    device=device,
                    dtype=val_labels.dtype
                )

                for c_id_tensor in common_classes:

                    c_id = c_id_tensor.item()

                    # ---------------------------------------------
                    # Validation samples class c
                    # ---------------------------------------------
                    r_idx = (
                        val_labels == c_id
                    ).nonzero(as_tuple=True)[0]

                    # ---------------------------------------------
                    # Synthetic samples class c
                    # ---------------------------------------------
                    s_idx = (
                        synth_labels == c_id
                    ).nonzero(as_tuple=True)[0]

                    if len(r_idx) == 0 or len(s_idx) == 0:
                        continue

                    # Loss từng sample của validation
                    real_loss_c = loss_real_each[r_idx]

                    # Loss từng sample của synthetic
                    synth_loss_c = loss_synth_each[s_idx]

                    # Pairwise difference:
                    # [g_i, 1] - [1, n_i]
                    pairwise_diff = torch.abs(
                        synth_loss_c[:, None]
                        -
                        real_loss_c[None, :]
                    )

                    eps_i = pairwise_diff.mean()

                    eps_class_losses.append(eps_i)

                # Average over common classes
                if len(eps_class_losses) > 0:

                    eps_loss = torch.stack(
                        eps_class_losses
                    ).mean()

                    val_loss += lambda_eps * eps_loss

                else:

                    eps_loss = torch.tensor(
                        0.0,
                        device=device
                    )

            # =====================================================
            # SAMPLE PROTOTYPES
            # =====================================================
            if use_l2 or use_l3:

                samples = build_text_distribution_samples(
                    mu,
                    kappa,
                    kappa_scale=kappa_scale
                )

            # =====================================================
            # REGULARIZATION
            # =====================================================
            reg_losses = []

            present_classes = val_labels.unique()

            if use_l1 or use_l3:

                present_classes = torch.cat(
                    [
                        val_labels,
                        synth_labels
                    ]
                ).unique()

            number_of_classes = len(present_classes)

            real_reg_loss_total = torch.tensor(
                0.0,
                device=device
            )

            synth_reg_loss_total = torch.tensor(
                0.0,
                device=device
            )

            # =====================================================
            # REG PER CLASS
            # =====================================================
            for c_id_tensor in present_classes:

                c_id = c_id_tensor.item()

                # -------------------------------------------------
                # REAL REG
                # -------------------------------------------------
                r_idx = (
                    val_labels == c_id
                ).nonzero(as_tuple=True)[0]

                if use_l2 and len(r_idx) > 0:

                    l_reg_r = compute_reg_vectorized(
                        logit_scale,
                        c_id,
                        samples,
                        val_imgs_embedding[r_idx],
                        mu
                    )

                    real_reg_loss_total += l_reg_r

                    reg_losses.append(
                        lamda2 *
                        l_reg_r /
                        number_of_classes
                    )

                # -------------------------------------------------
                # SYNTH REG
                # -------------------------------------------------
                if (
                    use_l3
                    and synth_imgs_embedding is not None
                ):

                    s_idx = (
                        synth_labels == c_id
                    ).nonzero(as_tuple=True)[0]

                    if len(s_idx) > 0:

                        l_reg_s = compute_reg_vectorized(
                            logit_scale,
                            c_id,
                            samples,
                            synth_imgs_embedding[s_idx],
                            mu
                        )

                        synth_reg_loss_total += l_reg_s

                        reg_losses.append(
                            lamda3 *
                            l_reg_s /
                            number_of_classes
                        )

            # =====================================================
            # ADD REGULARIZATION
            # =====================================================
            if reg_losses:

                val_loss += torch.sum(
                    torch.stack(reg_losses)
                )

            # =====================================================
            # ACCUMULATE LOSS
            # =====================================================
            total_val_loss += val_loss.item()

            total_real_loss += loss_real.item()

            if use_l1:
                total_synth_loss += loss_synth.item()

            if use_eps:
                total_eps_loss += eps_loss.item()

            if use_l2:
                total_real_reg_loss += (
                    real_reg_loss_total /
                    number_of_classes
                ).item()

            if use_l3:
                total_synth_reg_loss += (
                    synth_reg_loss_total /
                    number_of_classes
                ).item()

    # =========================================================
    # AVERAGE OVER VALIDATION BATCHES
    # =========================================================
    avg_val_loss = total_val_loss / max(num_batches, 1)
    avg_real_loss = total_real_loss / max(num_batches, 1)

    # =========================================================
    # LOGGING
    # =========================================================
    writer.add_scalar(
        "Validation Loss/Real_Base",
        avg_real_loss,
        step
    )

    if use_l1:

        writer.add_scalar(
            "Validation Loss/Synth_Base",
            total_synth_loss / max(num_batches, 1),
            step
        )

    if use_eps:

        writer.add_scalar(
            "Validation Loss/Eps",
            total_eps_loss / max(num_batches, 1),
            step
        )

    if use_l2:

        writer.add_scalar(
            "Validation Loss/Real_Reg",
            total_real_reg_loss / max(num_batches, 1),
            step
        )

    if use_l3:

        writer.add_scalar(
            "Validation Loss/Synth_Reg",
            total_synth_reg_loss / max(num_batches, 1),
            step
        )

    writer.add_scalar(
        "Validation Loss/Total",
        avg_val_loss,
        step
    )

    accuracy = correct / total

    return accuracy
#############################
def train_one_epoch_update(
    model,
    opt_h,
    scaler,
    step,
    fewshot_train_loader,
    loader_iter_G,
    lamda1,
    lamda2,
    lamda3,
    lambda_eps,
    use_l1,
    use_l2,
    use_l3,
    use_eps,
    writer,
    device,
    dataset="dtd",
    logit_scale=15,
    num_samples=30,
    kappa_scale=1.0
):

    model.model.train()

    for real_images, real_labels in tqdm(fewshot_train_loader):
        step += 1
        real_labels = real_labels.to(device)

        # =========================================================
        # REAL EMBEDDING
        # =========================================================
        real_imgs_embedding = get_image_embedding(
            model,
            real_images
        )
        # =========================================================
        # SYNTH
        # =========================================================
        if use_l1 or use_l3 or use_eps:
            synth_images, synth_labels = next(loader_iter_G)
            synth_labels = synth_labels.to(device)
            synth_imgs_embedding = get_image_embedding(
                model,
                synth_images
            )
        else:
            synth_imgs_embedding = None
            synth_labels = None
        # =========================================================
        # PROTOTYPE
        # =========================================================
        mu, kappa = get_mu_and_kappa(
            model,
            dataset
        )

        # =========================================================
        # BASE LOSS
        # =========================================================
        with torch.amp.autocast('cuda'):

            # -----------------------------------------------------
            # REAL
            # -----------------------------------------------------
            logits_real_all = (
                logit_scale *
                (real_imgs_embedding @ mu.t())
            )
            # Giữ loss từng sample
            loss_real_each = F.cross_entropy(
                logits_real_all,
                real_labels,
                reduction="none"
            )

            loss_real = loss_real_each.mean()
            total_loss = loss_real
            # -----------------------------------------------------
            # SYNTH
            # -----------------------------------------------------
            if use_l1:
                logits_synth_all = (
                    logit_scale *
                    (synth_imgs_embedding @ mu.t())
                )

                # Giữ loss từng sample
                loss_synth_each = F.cross_entropy(
                    logits_synth_all,
                    synth_labels,
                    reduction="none"
                )

                loss_synth = loss_synth_each.mean()
                total_loss += lamda1 * loss_synth

        # =========================================================
        # EPS
        #
        # eps_i =
        # 1 / (g_i * n_i)
        # sum_{g in G_i} sum_{s in S_i}
        # |l(g) - l(s)|
        # =========================================================

        if use_eps:
            eps_class_losses = []
            # Chỉ xét các class xuất hiện ở cả hai tập
            common_classes = torch.tensor(
                list(
                    set(real_labels.tolist())
                    &
                    set(synth_labels.tolist())
                ),
                device=device,
                dtype=real_labels.dtype
            )
            for c_id_tensor in common_classes:
                c_id = c_id_tensor.item()

                # -------------------------------------------------
                # Real samples class i
                # -------------------------------------------------
                r_idx = (
                    real_labels == c_id
                ).nonzero(as_tuple=True)[0]

                # -------------------------------------------------
                # Synth samples class i
                # -------------------------------------------------
                s_idx = (
                    synth_labels == c_id
                ).nonzero(as_tuple=True)[0]

                if len(r_idx) == 0 or len(s_idx) == 0:
                    continue
                # Loss của real class i
                real_loss_c = loss_real_each[r_idx]
                # Loss của synth class i
                synth_loss_c = loss_synth_each[s_idx]
                # -------------------------------------------------
                # Pairwise difference
                #
                # synth_loss_c: [g_i]
                # real_loss_c:  [n_i]
                #
                # sau broadcasting:
                # [g_i, 1] - [1, n_i]
                #
                # => [g_i, n_i]
                # -------------------------------------------------
                pairwise_diff = torch.abs(
                    synth_loss_c[:, None]
                    -
                    real_loss_c[None, :]
                )

                # -------------------------------------------------
                # eps_i
                # -------------------------------------------------
                eps_i = pairwise_diff.mean()
                eps_class_losses.append(eps_i)

            # -----------------------------------------------------
            # Average over classes
            # -----------------------------------------------------
            if len(eps_class_losses) > 0:
                eps_loss = torch.stack(
                    eps_class_losses
                ).mean()
                total_loss += lambda_eps * eps_loss

            else:
                eps_loss = torch.tensor(
                    0.0,
                    device=device
                )

        # =========================================================
        # SAMPLE PROTOTYPES
        # =========================================================
        if use_l2 or use_l3:
            samples = build_text_distribution_samples(
                mu,
                kappa,
                kappa_scale=kappa_scale
            )

        # =========================================================
        # REGULARIZATION
        # =========================================================

        reg_losses = []
        present_classes = real_labels.unique()
        if use_l1 or use_l3:
            present_classes = torch.cat(
                [
                    real_labels,
                    synth_labels
                ]
            ).unique()

        number_of_classes = len(present_classes)
        # Dùng để logging
        real_reg_loss_total = torch.tensor(
            0.0,
            device=device
        )

        synth_reg_loss_total = torch.tensor(
            0.0,
            device=device
        )

        # =========================================================
        # REG PER CLASS
        # =========================================================
        for c_id_tensor in present_classes:
            c_id = c_id_tensor.item()
            r_idx = (
                real_labels == c_id
            ).nonzero(as_tuple=True)[0]

            # -----------------------------------------------------
            # REG REAL
            # -----------------------------------------------------
            if use_l2 and len(r_idx) > 0:

                l_reg_r = compute_reg_vectorized(
                    logit_scale,
                    c_id,
                    samples,
                    real_imgs_embedding[r_idx],
                    mu
                )

                real_reg_loss_total += l_reg_r

                reg_losses.append(
                    lamda2 *
                    l_reg_r /
                    number_of_classes
                )

            # -----------------------------------------------------
            # REG SYNTH
            # -----------------------------------------------------
            if (
                use_l3
                and synth_imgs_embedding is not None
            ):

                s_idx = (
                    synth_labels == c_id
                ).nonzero(as_tuple=True)[0]

                if len(s_idx) > 0:

                    l_reg_s = compute_reg_vectorized(
                        logit_scale,
                        c_id,
                        samples,
                        synth_imgs_embedding[s_idx],
                        mu
                    )

                    synth_reg_loss_total += l_reg_s

                    reg_losses.append(
                        lamda3 *
                        l_reg_s /
                        number_of_classes
                    )

        # =========================================================
        # ADD REG LOSS
        # =========================================================

        if reg_losses:

            total_loss += torch.sum(
                torch.stack(reg_losses)
            )

        # =========================================================
        # OPTIMIZATION
        # =========================================================

        opt_h.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(opt_h)
        scaler.update()
        # =========================================================
        # LOGGING
        # =========================================================

        writer.add_scalar(
            "Training Loss/Real_Base",
            loss_real.item(),
            step
        )

        if use_l1:

            writer.add_scalar(
                "Training Loss/Synth_Base",
                loss_synth.item(),
                step
            )

        if use_eps:
            writer.add_scalar(
                "Training Loss/Eps",
                eps_loss.item(),
                step
            )

        if use_l2:

            writer.add_scalar(
                "Training Loss/Real_Reg",
                (
                    real_reg_loss_total /
                    number_of_classes
                ).item(),
                step
            )

        if use_l3:

            writer.add_scalar(
                "Training Loss/Synth_Reg",
                (
                    synth_reg_loss_total /
                    number_of_classes
                ).item(),
                step
            )

        writer.add_scalar(
            "Training Loss/Total",
            total_loss.item(),
            step
        )

    print(
        "Training Loss/Total",
        total_loss.item()
    )

    return step
#################################
def load_best_model(model, exp_name, ckpt_path, device="cuda"):
    import os
    import torch

    #best_model_path = os.path.join(ckpt_path, f"best_model_{exp_name}.pt")
    best_model_path = ckpt_path
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best_model.pt found in {ckpt_path}")

    checkpoint = torch.load(best_model_path, map_location=device)

    # ⚠️ vì bạn save model.model.state_dict()
    model.model.load_state_dict(checkpoint["model_state_dict"])

    model.model.to(device)
    model.model.eval()

    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best validation acc: {checkpoint['best_acc']*100:.2f}%")

    return model, checkpoint

########## refined LDA prompts ################
with open("/airc_lda_refined.json", "r") as f:
    loaded_list = json.load(f)

##################
@torch.no_grad()
def get_mu_and_kappa(model, dataset):
    all_mus =[]
    all_kappas =[]

    for class_idx in range(len(SUBSET_NAMES[dataset])):
        processed_text_inputs = [{"text": s} for s in loaded_list[class_idx]]
        class_embs = model.process(processed_text_inputs)
        class_embs = F.normalize(class_embs, dim=-1)#thừa

        mu = class_embs.mean(dim=0)
        D = mu.shape[0]
        R = mu.norm()

        mu = mu / R
        kappa = (R * (D - R**2)) / torch.clamp(1 - R**2, min=1e-6)

        all_mus.append(mu)
        all_kappas.append(kappa)

    all_mus = torch.stack(all_mus)
    all_kappas = torch.stack(all_kappas)

    return all_mus, all_kappas

"""# Training"""

dataset = 'fgvc_aircraft'
n_samples_per_class = 16
n_synth_per_class = 64
n_val_per_class = 64
n_epochs = 45
batch_size = 64
eval_batch_size = 128
logit_scale = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "qwen"
synth_train_data_dir = "/workspace/airc_synth/content/synthetic_regions/Medium"
fix_random_seed(0)
# Shared lambda
lambda1, lambda2, lambda3,  lambda_eps, kappa_scale = 0.2, 0.04, 0.02, 0.1, 1.0
num_samples_area = [30]

##################################3
fewshot_train_loader,  test_loader = get_data_loader(
        real_train_data_dir='/data/',
        real_test_data_dir="/data/",
        dataset=dataset,
        bs=batch_size,
        eval_bs=eval_batch_size,
        n_img_per_cls=n_samples_per_class,
        model_type=model_type,
    )

synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=synth_train_data_dir,
        bs=batch_size,
        n_img_per_cls=n_synth_per_class,
        dataset=dataset,
        model_type=model_type )
print(f"Number of few-shot training samples: {len(fewshot_train_loader.dataset)}")
print(f"Number of synthetic training samples: {len(synth_train_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

for num_samples in num_samples_area:

    print("\n" + "="*60)
    print(f"Number of prototype sampled per area: {num_samples}")
    print("="*60)

    use_l1 = True
    use_l2 = True
    use_l3 = True
    use_eps = True

    exp_name = f"{dataset}_{lambda1}_{lambda2}_{lambda3}_{lambda_eps}_{kappa_scale}"
    log_dir_path = f"/data/runs/{exp_name}"
    ckpt_path = f"/data/checkpoints/{exp_name}"

    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    # ===== RESET MODEL =====
    model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv", "proj", "q_proj", "v_proj"], #q_proj, v_proj: Text Encoder
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    model.model = get_peft_model(model.model, lora_config)
    model.model.gradient_checkpointing_enable() # Commenting this out to fix CheckpointError

    writer = SummaryWriter(log_dir=log_dir_path)

    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    opt_h = torch.optim.AdamW(trainable_params, lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    step = 0
    best_acc = 0.0

    loader_iter_G = cycle(synth_train_loader)

    for epoch in range(n_epochs):
        step = train_one_epoch_update(
            model=model,
            opt_h=opt_h,
            scaler=scaler,
            step=step,
            fewshot_train_loader=fewshot_train_loader,
            loader_iter_G=loader_iter_G,
            lamda1=lambda1,
            lamda2=lambda2,
            lamda3=lambda3,
            lambda_eps = lambda_eps,
            use_l1=use_l1,
            use_l2=use_l2,
            use_l3=use_l3,
            use_eps = use_eps,
            writer=writer,
            device=device,
            dataset=dataset,
            logit_scale=logit_scale,
            num_samples = num_samples,
            kappa_scale = kappa_scale
        )

        model.model.eval()
        with torch.no_grad():
                mu_eval, _ = get_mu_and_kappa(model, dataset)

                mu_eval = mu_eval.to(device)
                test_acc = get_val_losses(
     model=model,
     opt_h=opt_h,
     scaler=scaler,
     step=step,
     val_loader = test_loader,
     lamda1=lambda1,
     lamda2=lambda2,
     lamda3=lambda3,
     lambda_eps = lambda_eps,
     use_l1=use_l1,
     use_l2=use_l2,
     use_l3=use_l3,
     use_eps = use_eps,
     writer=writer,
     device=device,
     loader_iter_G=loader_iter_G,
     dataset=dataset,
     logit_scale=15,
     num_samples=30,
     kappa_scale=1.0
)
                print(f"[{num_samples}] Epoch {epoch} | Test ACC {test_acc*100:.2f}%")
                writer.add_scalar("Eval/Accuracy", test_acc, epoch)

                #  SAVE BEST MODEL
                if test_acc > best_acc:
                    best_acc = test_acc
                    save_path = os.path.join(ckpt_path, f"best_model_{exp_name}.pt")

                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.model.state_dict(),
                        "best_acc": best_acc,
                        "lamda1": lambda1,
                        "lamda2": lambda2,
                        "lamda3": lambda3,
                    }, save_path)

                    print(f"Saved BEST model at epoch {epoch} with eval acc = {best_acc*100:.2f}")
    writer.close()
    print(f"✅ DONE| Best Test Acc: {best_acc*100:.2f}%")


