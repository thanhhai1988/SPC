
import sys
import os
import shutil
import random
import gc
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
    writer,
    device,
    dataset="dtd",
    logit_scale=15
):
    model.model.train()

    for real_images, real_labels in tqdm(fewshot_train_loader):
        step += 1
        synth_images, synth_labels = next(loader_iter_G)
        real_labels = real_labels.to(device)
        synth_labels = synth_labels.to(device)
        mu, kappa = get_mu_and_kappa(model, dataset)
        torch.cuda.empty_cache()
        with torch.amp.autocast('cuda'):
            real_imgs_embedding = get_image_embedding(model, real_images)
            synth_imgs_embedding = get_image_embedding(model, synth_images)
            logits_real_all = logit_scale * (real_imgs_embedding @ mu.t())
            logits_synth_all = logit_scale * (synth_imgs_embedding @ mu.t())

            samples = build_text_distribution_samples(mu)

        log_metrics = {"br": 0, "rr": 0, "bs": 0, "rs": 0}

        with torch.amp.autocast('cuda'):
            loss_real = F.cross_entropy(logits_real_all, real_labels)
            loss_synth = F.cross_entropy(logits_synth_all, synth_labels)

            total_loss = loss_real + lamda1 * loss_synth

            log_metrics["br"] = loss_real.item()
            log_metrics["bs"] = loss_synth.item()

        reg_losses =[]
        log_rr, log_rs = 0, 0

        present_classes = torch.cat([real_labels, synth_labels]).unique()
        number_of_classes = len(present_classes)

        with torch.amp.autocast('cuda'):
            for c_id_tensor in present_classes:
                c_id = c_id_tensor.item()

                r_idx = (real_labels == c_id).nonzero(as_tuple=True)[0]
                s_idx = (synth_labels == c_id).nonzero(as_tuple=True)[0]

                if len(r_idx) > 0:
                    l_reg_r = compute_reg_vectorized(
                        logit_scale=logit_scale,
                        area_index=c_id,
                        samples=samples,
                        feats_i=real_imgs_embedding[r_idx],
                        centroids=mu
                    )
                    reg_losses.append(lamda2 * l_reg_r / number_of_classes)
                    log_rr += l_reg_r.item() / number_of_classes

                if len(s_idx) > 0:
                    l_reg_s = compute_reg_vectorized(
                        logit_scale=logit_scale,
                        area_index=c_id,
                        samples=samples,
                        feats_i=synth_imgs_embedding[s_idx],
                        centroids=mu
                    )
                    reg_losses.append(lamda3 * l_reg_s / number_of_classes)
                    log_rs += l_reg_s.item() / number_of_classes

            if reg_losses:
                total_loss = total_loss + torch.sum(torch.stack(reg_losses))
        print(f'Total loss {total_loss.item()}')
        opt_h.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(opt_h)
        scaler.update()

        writer.add_scalar("Loss_Batch/Real_Base", log_metrics["br"], step)
        writer.add_scalar("Loss_Batch/Real_Reg", log_rr, step)
        writer.add_scalar("Loss_Batch/Synth_Base", log_metrics["bs"], step)
        writer.add_scalar("Loss_Batch/Synth_Reg", log_rs, step)
        writer.add_scalar("Loss_Batch/Total", total_loss.item(), step)

        del total_loss, logits_real_all, logits_synth_all, samples

    return step

def get_val_losses(
    model,
    opt_h,
    scaler,
    step,
    val_loader,
    lamda1,
    lamda2,
    lamda3,
    lamda4,
    use_l1,
    use_l2,
    use_l3,
    use_l4,
    writer,
    device,
    loader_iter_G = None,
    dataset="dtd",
    logit_scale=15,
    num_samples= 30
):
    model.model.eval()
    correct = 0
    total = 0

    for val_images, val_labels in tqdm(val_loader):
        val_labels = val_labels.to(device)

        # ===== REAL EMBEDDING =====
        val_imgs_embedding = get_image_embedding(model, val_images)

        # ===== PROTOTYPE =====
        mu, kappa = get_mu_and_kappa(model, dataset)

        # ACC
        similarity = logit_scale * (val_imgs_embedding  @ mu.T)
        preds = similarity.argmax(dim=1)
        correct += (preds == val_labels).sum().item()
        total += val_labels.size(0)

        with torch.amp.autocast('cuda'):

            logits_real_all = logit_scale * (val_imgs_embedding @ mu.t())
            loss_real = F.cross_entropy(logits_real_all, val_labels)

            val_loss = loss_real

        # ===== SAMPLE PROTOTYPES =====
        if use_l2 or use_l3:
            samples = build_text_distribution_samples(mu, kappa)

        reg_losses = []
        present_classes = val_labels.unique()
        number_of_classes = len(present_classes)

        # ===== REGULARIZATION =====
        for c_id_tensor in present_classes:
            c_id = c_id_tensor.item()

            r_idx = (val_labels == c_id).nonzero(as_tuple=True)[0]

            # ---- REG REAL ----
            if use_l2 and len(r_idx) > 0:
                l_reg_r = compute_reg_vectorized(
                    logit_scale,
                    c_id,
                    samples,
                    val_imgs_embedding[r_idx],
                    mu
                )
                reg_losses.append(lamda2 * l_reg_r / number_of_classes)

        #separate loss
        if use_l4:
          sep_loss =  vmf_kl_approx(mu, kappa)
          val_loss += lamda4 * sep_loss
        if reg_losses:
            val_loss += torch.sum(torch.stack(reg_losses))

        writer.add_scalar("Validation Loss/Real_Base", loss_real.item(), step)

        if use_l2:
            writer.add_scalar("Validation Loss/Real_Reg", (l_reg_r / number_of_classes).item(), step)

        if use_l4:
            writer.add_scalar("Validation Loss /Separation_Loss", sep_loss.item(), step)
        writer.add_scalar("Validation Loss/Total", val_loss.item(), step)
    return correct / total

def train_one_epoch_update(
    model,
    opt_h,
    scaler,
    step,
    fewshot_train_loader,
    lamda1,
    lamda2,
    lamda3,
    lamda4,
    use_l1,
    use_l2,
    use_l3,
    use_l4,
    writer,
    device,
    loader_iter_G = None,
    dataset="dtd",
    logit_scale=15,
    num_samples= 30
):
    model.model.train()

    for real_images, real_labels in tqdm(fewshot_train_loader):

        step += 1
        real_labels = real_labels.to(device)

        # ===== REAL EMBEDDING =====
        real_imgs_embedding = get_image_embedding(model, real_images)

        # ===== SYNTH (chỉ load khi cần) =====
        if use_l1 or use_l3:
            synth_images, synth_labels = next(loader_iter_G)
            synth_labels = synth_labels.to(device)
            synth_imgs_embedding = get_image_embedding(model, synth_images)
        else:
            synth_imgs_embedding = None

        # ===== PROTOTYPE =====
        mu, kappa = get_mu_and_kappa(model, dataset)

        with torch.amp.autocast('cuda'):

            logits_real_all = logit_scale * (real_imgs_embedding @ mu.t())
            loss_real = F.cross_entropy(logits_real_all, real_labels)

            total_loss = loss_real

            # ===== SYNTH LOSS =====
            if use_l1:
                logits_synth_all = logit_scale * (synth_imgs_embedding @ mu.t())
                loss_synth = F.cross_entropy(logits_synth_all, synth_labels)
                total_loss += lamda1 * loss_synth

        # ===== SAMPLE PROTOTYPES =====
        if use_l2 or use_l3:
            samples = build_text_distribution_samples(mu, kappa, kappa_scale= 0.05)

        reg_losses = []
        present_classes = real_labels.unique()

        if use_l1 or use_l3:
            present_classes = torch.cat([real_labels, synth_labels]).unique()

        number_of_classes = len(present_classes)

        # ===== REGULARIZATION =====
        for c_id_tensor in present_classes:
            c_id = c_id_tensor.item()

            r_idx = (real_labels == c_id).nonzero(as_tuple=True)[0]

            # ---- REG REAL ----
            if use_l2 and len(r_idx) > 0:
                l_reg_r = compute_reg_vectorized(
                    logit_scale,
                    c_id,
                    samples,
                    real_imgs_embedding[r_idx],
                    mu
                )
                reg_losses.append(lamda2 * l_reg_r / number_of_classes)

            # ---- REG SYNTH ----
            if use_l3 and synth_imgs_embedding is not None:
                s_idx = (synth_labels == c_id).nonzero(as_tuple=True)[0]

                if len(s_idx) > 0:
                    l_reg_s = compute_reg_vectorized(
                        logit_scale,
                        c_id,
                        samples,
                        synth_imgs_embedding[s_idx],
                        mu
                    )
                    reg_losses.append(lamda3 * l_reg_s / number_of_classes)
        #separate loss
        #if use_l4:
        #  sep_loss =  vmf_kl_approx(mu, kappa)
        #  total_loss += lamda4 * sep_loss
        if reg_losses:
            total_loss += torch.sum(torch.stack(reg_losses))

        # ===== OPTIM =====
        opt_h.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(opt_h)
        scaler.update()

        writer.add_scalar("Training Loss/Real_Base", loss_real.item(), step)

        if use_l1:
            writer.add_scalar("Training Loss/Synth_Base", loss_synth.item(), step)
        if use_l2:
            writer.add_scalar("Training Loss/Real_Reg", (l_reg_r / number_of_classes).item(), step)
        if use_l3:
            writer.add_scalar("Training Loss/Synth_Reg", (l_reg_s / number_of_classes).item(), step)
        #if use_l4:
        #    writer.add_scalar("Training Loss/Separation_Loss", sep_loss.item(), step)
        writer.add_scalar("Training Loss/Total", total_loss.item(), step)
    print("Training Loss/Total", total_loss.item())
    return step

def load_best_model(model, exp_name, ckpt_path, device="cuda"):
    import os
    import torch

    best_model_path = os.path.join(ckpt_path, f"best_model_{exp_name}.pt")

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



import json
#Experiment 1: prompts chỉ dựa trên text: tên nhãn lớp; tạo prompts có tính phân biệt giữa các tên lớp được cho trước
with open("/airc_lda_refined.json", "r") as f:
    loaded_list = json.load(f)



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
########################## Tunning###############
dataset = 'fgvc_aircraft'
n_samples_per_class = 16
n_synth_per_class = 64
n_epochs = 1
batch_size = 64
eval_batch_size = 128
logit_scale = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "qwen"
#synth_train_data_dir = "/content/synth_data/data_eurosat/data_eurosat"
fix_random_seed(0)
# ================= ABLATION SETTINGS =================
ablation_settings = [
    #{"name": "base", "use_l1": False, "use_l2": False, "use_l3": False},
    #{"name": "synthetic", "use_l1": True, "use_l2": False, "use_l3": False},
    {"name": "reg_real", "use_l1": True, "use_l2": True, "use_l3": False},
    #{"name": "reg_synth", "use_l1": True, "use_l2": False, "use_l3": True},
    #{"name": "full", "use_l1": True, "use_l2": True, "use_l3": True},
]
ablation_acc = [
     #{"name": "base", "best ACC": 0},
     #{"name": "synthetic",  "best ACC": 0},
     {"name": "reg_real",  "best ACC": 0},
     #{"name": "reg_synth",  "best ACC": 0},
     #{"name": "full",  "best ACC": 0},
]

# Shared lambda
lambda1, lambda2, lambda3, lambda4 = 0.2, 0.04, 0.02, 0.5
# number samples each area
num_samples_area = [30]
#K_acc = {5: None, 10: None, 30: None, 100: None }
K_acc = {30: None}
# Setting num_workers to 2 to address the DataLoader warning
fewshot_train_loader,  test_loader = get_data_loader(
        real_train_data_dir='/data/',
        real_test_data_dir="/data/",
        dataset=dataset,
        bs=batch_size,
        eval_bs=eval_batch_size,
        n_img_per_cls=n_samples_per_class,
        model_type=model_type,
    )

print(f"Number of few-shot training samples: {len(fewshot_train_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

for num_samples in num_samples_area:

    print("\n" + "="*60)
    print(f"Number of prototype sampled per area: {num_samples}")
    print("="*60)

    use_l1 = False
    use_l2 = False
    use_l3 = False
    use_l4 = False

    exp_name = f"{dataset}_real_finetune"
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

    #loader_iter_G = cycle(synth_train_loader)

    for epoch in range(n_epochs):
        step = train_one_epoch_update(
            model=model,
            opt_h=opt_h,
            scaler=scaler,
            step=step,
            fewshot_train_loader=fewshot_train_loader,
            #loader_iter_G=loader_iter_G,
            lamda1=lambda1,
            lamda2=lambda2,
            lamda3=lambda3,
            lamda4=lambda4,
            use_l1=use_l1,
            use_l2=use_l2,
            use_l3=use_l3,
            use_l4=use_l4,
            writer=writer,
            device=device,
            dataset=dataset,
            logit_scale=logit_scale,
            num_samples = num_samples
        )

        model.model.eval()
        with torch.no_grad():
                test_acc = get_val_losses(
            model,
            opt_h,
            scaler,
            epoch,
            test_loader,
            #loader_iter_G,
            lamda1=lambda1,
            lamda2=lambda2,
            lamda3=lambda3,
            lamda4=lambda4,
            use_l1=use_l1,
            use_l2=use_l2,
            use_l3=use_l3,
            use_l4=use_l4,
            writer=writer,
            device=device,
            dataset=dataset,
            logit_scale=logit_scale,
            num_samples = num_samples
        )
                print(f"[{num_samples}] Epoch {epoch} | Eval ACC {test_acc*100:.2f}%")

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
    #print(f"✅ DONE K={num_samples} | Best Evak Acc: {best_acc*100:.2f}%")
    #model, ckpt = load_best_model(model, exp_name, ckpt_path, device)

    #mu_eval, kappa = get_mu_and_kappa(model, dataset)

    #mu_eval = mu_eval.to(device)
    #test_acc = get_acc(model, test_loader, mu_eval, logit_scale, device)
    print(f"Test accuracy (best model) with K = {num_samples}: {best_acc*100:.2f}%")
    #K_acc[num_samples] = test_acc
    #print("="*60)
#print(K_acc)
print('Done')

