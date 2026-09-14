
import sys
import os
import shutil
import random
import gc
import faiss
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_train import *
from itertools import cycle
from tqdm import tqdm
import numpy as np
from util_data import SUBSET_NAMES, TEMPLATES_SMALL
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

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


########################################


dataset = 'eurosat'
classes = SUBSET_NAMES[dataset]
n_samples_per_class = 16
n_synth_per_class = 64
n_epochs = 40
batch_size = 64
eval_batch_size = 128
logit_scale = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "qwen"
synth_train_data_dir = "/synth_data/data_eurosat/data_eurosat"
fix_random_seed(0)

# Shared lambda
lambda1, lambda2, lambda3, lambda4 = 0.2, 0.04, 0.02, 0.5

# Setting num_workers to 2 to address the DataLoader warning
fewshot_train_loader, test_loader = get_data_loader(
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


########################################
all_vectors = []
for images, labels in fewshot_train_loader:
    # images là list PIL images
    for img in images:
        arr = np.array(img, dtype=np.float32)
        vec = arr.reshape(-1)
        all_vectors.append(vec)

for images, labels in synth_train_loader:
    # images là list PIL images
    for img in images:
        arr = np.array(img, dtype=np.float32)
        vec = arr.reshape(-1)
        all_vectors.append(vec)

X = np.stack(all_vectors).astype(np.float32)

print(X.shape)

n_class = len(classes)

#######################################
D = X.shape[1]
K = n_class*2 #number of clusters 
kmeans = faiss.Kmeans(d = D, k = K)
kmeans.train(X)

index = kmeans.index

def check_area(X_batch):
    # check area of a PIL image
    I = index.search(X_batch, 1)
    return I[1]

def eps_area_GS(TS, idx_cls, L_real, L_syn):
    """
    Compute eps_i = eps(Si, Gi):
    mean L2 distance between real and synthetic logits
    in area idx_cls.

    L_real: (N_real, C)
    L_syn : (N_syn, C)
    """
    L_real_area = L_real[TS[idx_cls][0]]   # (Nr, C)
    L_syn_area  = L_syn[TS[idx_cls][1]]    # (Ns, C)

    diff = torch.norm(
        L_real_area[:, None, :] - L_syn_area[None, :, :],
        p=2,
        dim=-1
    )  # (Nr, Ns)

    eps_i = diff.mean()
    return eps_i


def eps_area_GZ(TS, idx_cls, L_syn):
    """
    Compute eps_i_GZ = eps(Gi, Zi):
    mean pairwise L2 distance between synthetic logits
    in area idx_cls.

    L_syn : (N_syn, C)
    """
    L_syn_area = L_syn[TS[idx_cls][1]]     # (Ns, C)

    diff = torch.norm(
        L_syn_area[:, None, :] - L_syn_area[None, :, :],
        p=2,
        dim=-1
    )  # (Ns, Ns)

    eps_GZ_i = diff.mean()
    return eps_GZ_i

def epsilon(TS, L_real, L_syn):
    """
    Compute loss h elements: eps_GS, eps_GZ,
    g: total syn points in area TS
    """
    eps_GS = torch.tensor(0.0, device=device)
    eps_GZ = torch.tensor(0.0, device=device)
    g = 0
    for idx_clus in range(K):
        if (len(TS[idx_clus][0]) != 0) & (len(TS[idx_clus][1]) != 0):
            eps_GS += eps_area_GS(TS, idx_clus, L_real, L_syn)*len(TS[idx_clus][1])
            g += len(TS[idx_clus][1])
            eps_GZ += eps_area_GZ(TS, idx_clus, L_syn)*len(TS[idx_clus][1])
    eps_GS = eps_GS / g
    eps_GZ = eps_GZ / g
    if g == 0:
        eps_GS = torch.tensor(0.0, device=device)
        eps_GZ = torch.tensor(0.0, device=device)
    return eps_GS, eps_GZ

def train_one_epoch_h(model, epoch, step, lamda, lamda1, lamda2, logit_scale = 15):
    for real_images, real_labels in fewshot_train_loader:
        step += 1
        mu, kappa = get_mu_and_kappa(model, dataset)

        synth_images, synth_labels = next(loader_iter_G) # Renamed for clarity

        # Move original labels to device for potential augmentation later
        real_labels = real_labels.to(device)
        synth_labels = synth_labels.to(device)

        mu, kappa = get_mu_and_kappa(model, dataset)

        torch.cuda.empty_cache()
        with torch.amp.autocast('cuda'):

            real_imgs_embedding = get_image_embedding(model, real_images)
            synth_imgs_embedding = get_image_embedding(model, synth_images)
            logits_real_all = logit_scale * (real_imgs_embedding @ mu.t())
            logits_synth_all = logit_scale * (synth_imgs_embedding @ mu.t())

            # Use augmented labels for loss calculation
            real_loss = F.cross_entropy(logits_real_all, real_labels)
            synth_loss = F.cross_entropy(logits_synth_all, synth_labels)

            L_real_vector = logits_real_all
            L_syn_vector = logits_synth_all

        TS = {i: [[], []] for i in range(K)}

        # Use original PIL images for check_area, converting them to numpy as originally intended
        batch_real = np.stack(
            [np.asarray(img, dtype=np.float32).reshape(-1) for img in real_images]
        ).astype(np.float32)
        batch_synth = np.stack(
            [np.asarray(img, dtype=np.float32).reshape(-1) for img in synth_images]
        ).astype(np.float32)

        # Process all images in the batch at once for check_area
        real_areas_indices = check_area(batch_real) # check_area returns only indices
        synth_areas_indices = check_area(batch_synth)

        for i in range(batch_real.shape[0]):
            TS[real_areas_indices[i].item()][0].append(i) # sample ith in area real_areas[i]

        for i in range(batch_synth.shape[0]):
            TS[synth_areas_indices[i].item()][1].append(i) #sample i in area synth_areas[i]


        with torch.amp.autocast('cuda'):
            eps_GS, eps_GZ = epsilon(TS, L_real_vector, L_syn_vector)

            L_h = synth_loss + lamda * real_loss + lamda1 * eps_GS + lamda2 * eps_GZ

        writer.add_scalar("Classifier Loss",L_h.item(), step)
        writer.add_scalar("Real Loss", real_loss.item(), step)
        writer.add_scalar("Synth Loss",synth_loss.item(), step)
        writer.add_scalar("Eps GS",eps_GS.item(), step)
        writer.add_scalar("Eps GZ",eps_GZ.item(), step)
        print(f'Step {step} epoch {epoch} Loss h: ', L_h.item())
        opt_h.zero_grad()
        scaler.scale(L_h).backward()
        scaler.step(opt_h)
        scaler.update()
        gc.collect()
        torch.cuda.empty_cache()
    return step

"""# Training"""

dataset = 'eurosat'
n_samples_per_class = 16
n_synth_per_class = 64
n_val_per_class = 64
n_epochs = 40
batch_size = 32
eval_batch_size = 128
logit_scale = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "qwen"
synth_train_data_dir = "/data/synth_data/data_eurosat/data_eurosat"
fix_random_seed(0)


# Shared lambda
lamda, lamda1, lamda2  = 10, 0.5, 5# number samples each area

# Setting num_workers to 2 to address the DataLoader warning
fewshot_train_loader, test_loader = get_data_loader(
        real_train_data_dir='/content/',
        real_test_data_dir="/content/",
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



exp_name = f"{dataset}_proto_aug_{lamda}_{lamda1}_{lamda2}"
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
        step = train_one_epoch_h(model, epoch, step,
                                 lamda, lamda1, lamda2, logit_scale = 15)

        model.model.eval()
        with torch.no_grad():
                mu_eval, _ = get_mu_and_kappa(model, dataset)

                mu_eval = mu_eval.to(device)
                test_acc = get_acc(model, test_loader, mu_eval, logit_scale, device)
                print(f" Epoch {epoch} | Eval ACC {test_acc*100:.2f}%")
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

                    print(f"Saved BEST model at epoch {epoch} with test acc = {best_acc*100:.2f}")
writer.close()
print(f"✅ DONE  | Best Test Acc: {best_acc*100:.2f}%")

