import sys
sys.path.append('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_optimization')
import torch
from flow_train_loc import (
    TamboDataModule, 
    TamboDiffusionLitModel, 
    df_cleaned
)
from models.DiffusionCondition import DDIMSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1. Load trained checkpoint
# -------------------------------
ckpt_path = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_ckpts/diffusion/ckpt_epoch=1999.ckpt"

lit_model = TamboDiffusionLitModel.load_from_checkpoint(
    ckpt_path,
    map_location=device,
).to(device)
lit_model.eval()

model = lit_model.model
model.eval()

# -------------------------------
# 2. Build DDIM Sampler
# -------------------------------
sampler = DDIMSampler(
    model,
    beta_1=lit_model.hparams.beta_1,
    beta_T=lit_model.hparams.beta_T,
    T=lit_model.hparams.T,
    eta=0.0,
    ddim_steps=100,
)

# -------------------------------
# 3. Load test set
# -------------------------------
data_module = TamboDataModule(
    df=df_cleaned,
    batch_size=64,
    train_ratio=0.85,
    val_ratio=0.10,
    test_ratio=0.05,
    num_workers=4,
    seed=42,
)
data_module.setup("test")

test_loader = data_module.test_dataloader()

# -------------------------------
# 4. Extract FIRST 5 test samples
# -------------------------------


first5_imgs = []
first5_conds = []
count = 0

for imgs, cond in test_loader:
    energy, sin_z, cos_z, sin_a, cos_a = cond
    B = energy.size(0)

    for i in range(B):

        # Store image (3,32,32)
        first5_imgs.append(imgs[i].clone())

        # Store condition (5,)
        first5_conds.append(torch.stack([
            energy[i], sin_z[i], cos_z[i], sin_a[i], cos_a[i]
        ]))

        count += 1
        if count == 10:
            break

    if count == 10:
        break



# first5 is now a list of 5 conditioning vectors (each shape (5,))
print("Collected first 5 conditioning vectors.")

# -------------------------------
# 5. Generate 1000 samples per each conditioning
# -------------------------------
num_samples_total = 1000
chunk_size = 200  # try 16 or 8 if still OOM

generated_sets = []

for idx, cond_vec in enumerate(first5_conds):
    cond_vec = cond_vec.to(device)  # (5,)

    images_chunks = []
    samples_done = 0

    while samples_done < num_samples_total:
        bs = min(chunk_size, num_samples_total - samples_done)

        # noise: (bs, 3, 32, 32)
        noise = torch.randn(bs, 3, 32, 32, device=device)

        # (bs, 5) condition batch
        cond_batch = cond_vec.view(1, 5).expand(bs, 5)

        with torch.no_grad():
            gen_imgs_chunk = sampler(
                noise,
                cond_batch[:, 0],  # energy
                cond_batch[:, 1],  # sin z
                cond_batch[:, 2],  # cos z
                cond_batch[:, 3],  # sin a
                cond_batch[:, 4],  # cos a
            )  # (bs, 3, 32, 32)

        images_chunks.append(gen_imgs_chunk.cpu())

        # Free GPU memory for this chunk
        del noise, cond_batch, gen_imgs_chunk
        torch.cuda.empty_cache()

        samples_done += bs
        print(f"Condition {idx+1}/10: {samples_done}/{num_samples_total} samples done")

    # Concatenate all chunks -> (1000, 3, 32, 32)
    gen_imgs_all = torch.cat(images_chunks, dim=0)
    generated_sets.append({
        "condition": cond_vec.cpu(),
        "images": gen_imgs_all,   # (1000, 3, 32, 32)
    })

    # extra safety between conditions
    torch.cuda.empty_cache()
    print(f"Finished condition {idx+1}/5")

print("✔ Done: generated 5 × 1000 = 5000 images.")


import torch
import matplotlib.pyplot as plt
import numpy as np

num_conds = 10 # first 5 conditions

for idx in range(num_conds):
    # ----------------------------------------------------
    # 1. Ground truth image for this condition
    # ----------------------------------------------------
    gt_img = first5_imgs[idx].cpu().numpy()   # (3, H, W)

    gt_R = gt_img[0]
    gt_G = gt_img[1]
    gt_B = gt_img[2]

    gt_R_x, gt_R_y = gt_R.mean(axis=0), gt_R.mean(axis=1)
    gt_G_x, gt_G_y = gt_G.mean(axis=0), gt_G.mean(axis=1)
    gt_B_x, gt_B_y = gt_B.mean(axis=0), gt_B.mean(axis=1)

    # ----------------------------------------------------
    # 2. Generated images for this condition
    # ----------------------------------------------------
    gen_imgs = generated_sets[idx]["images"].cpu().numpy()  # (1000, 3, H, W)

    R_batch = gen_imgs[:, 0]   # (1000, H, W)
    G_batch = gen_imgs[:, 1]
    B_batch = gen_imgs[:, 2]

    R_x_mean = R_batch.mean(axis=1).mean(axis=0)
    R_y_mean = R_batch.mean(axis=2).mean(axis=0)

    G_x_mean = G_batch.mean(axis=1).mean(axis=0)
    G_y_mean = G_batch.mean(axis=2).mean(axis=0)

    B_x_mean = B_batch.mean(axis=1).mean(axis=0)
    B_y_mean = B_batch.mean(axis=2).mean(axis=0)

    # ----------------------------------------------------
    # 3. Condition values (for title)
    # ----------------------------------------------------
    cond_vec = generated_sets[idx]["condition"].cpu().numpy()  # (5,)
    energy, sin_z, cos_z, sin_a, cos_a = cond_vec.tolist()

    # ----------------------------------------------------
    # 4. Plot: 1 row, 6 columns
    # ----------------------------------------------------
    fig, axs = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle(
        f"Condition {idx+1}: "
        f"E={energy:.3f}, sin_z={sin_z:.3f}, cos_z={cos_z:.3f}, "
        f"sin_a={sin_a:.3f}, cos_a={cos_a:.3f}",
        y=1.05,
    )

    # --- R-X ---
    axs[0].plot(gt_R_x, label="GT")
    axs[0].plot(R_x_mean, label="Gen Mean", linestyle="--")
    axs[0].set_title("Particle Density vs x")

    # --- R-Y ---
    axs[1].plot(gt_R_y, label="GT")
    axs[1].plot(R_y_mean, label="Gen Mean", linestyle="--")
    axs[1].set_title("Particle Density vs y")

    # --- G-X ---
    axs[2].plot(gt_G_x, label="GT")
    axs[2].plot(G_x_mean, label="Gen Mean", linestyle="--")
    axs[2].set_title("Average arrival time vs x ")

    # --- G-Y ---
    axs[3].plot(gt_G_y, label="GT")
    axs[3].plot(G_y_mean, label="Gen Mean", linestyle="--")
    axs[3].set_title("Average arrival time vs y")

    # --- B-X ---
    axs[4].plot(gt_B_x, label="GT")
    axs[4].plot(B_x_mean, label="Gen Mean", linestyle="--")
    axs[4].set_title("Average arrival time vs x")

    # --- B-Y ---
    axs[5].plot(gt_B_y, label="GT")
    axs[5].plot(B_y_mean, label="Gen Mean", linestyle="--")
    axs[5].set_title("Average Kinetic Energy vs y")

    # One legend is enough
    axs[0].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"condition_{idx+1}.png", dpi=300, bbox_inches="tight")
    plt.close()

