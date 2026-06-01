#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:12:32 2026

@author: dimi
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==================================================
# settings
# ==================================================
num_augmentations = 15
seeds = [1,2,11,110, 210,310]   # <- put your seeds here
seeds = [1,2,410]
base_dir = "/Volumes/KINGSTON/geometric_more/convergence"
folder_template = (
    "geodesicGeodesic_augmentation_LC_noise_0.25_obs_dens_280_sim_time_500_seed_{seed}"
)

save_plot_dir = "/Volumes/KINGSTON/geometric_more/convergence/"
save_name = "performance_vs_augments_avg_over_seeds280"

# ==================================================
# helper function: read one seed
# ==================================================
def read_one_seed(seed, num_augmentations, base_dir, folder_template):
    """
    Reads one seed folder and returns:
        log_like : shape (num_augmentations,)
        wRMSEs   : shape (num_augmentations,)
        strengths
    Missing files are left as np.nan.
    """
    save_dir = os.path.join(base_dir, folder_template.format(seed=seed))

    log_like = np.full(num_augmentations, np.nan)
    wRMSEs = np.full(num_augmentations, np.nan)
    strengths = None

    # ----------------------------
    # initial file
    # ----------------------------
    init_file = os.path.join(save_dir, "Init_f_est.dat")
    print(f"\nseed {seed}")
    print("checking:", init_file)

    if os.path.exists(init_file):
        with open(init_file, "rb") as filehandler:
            data_init = pickle.load(filehandler)

        # initial value goes to augmentation index 0
        log_like[0] = data_init.get("ll_init", np.nan)
        wRMSEs[0] = data_init.get("wRMSE_init", np.nan)
    else:
        print(f"warning: missing {init_file}")

    # ----------------------------
    # augmentation files
    # ----------------------------
    for augmentation_num in range(num_augmentations):
        fname = os.path.join(
            save_dir,
            f"f_est_after_{augmentation_num}_augm_n_eval.dat"
        )

        if not os.path.exists(fname):
            print(f"warning: missing {fname}")
            continue

        with open(fname, "rb") as filehandler:
            data_aug = pickle.load(filehandler)

        log_like[augmentation_num] = data_aug.get("ll", np.nan)
        wRMSEs[augmentation_num] = data_aug.get("wRMSE_1", np.nan)

        if strengths is None and "strengths" in data_aug:
            strengths = data_aug["strengths"]

    return log_like, wRMSEs, strengths


# ==================================================
# read all seeds
# ==================================================
all_log_like = []
all_wRMSEs = []
all_strengths = []

for seed in seeds:
    log_like, wRMSEs, strengths = read_one_seed(
        seed=seed,
        num_augmentations=num_augmentations,
        base_dir=base_dir,
        folder_template=folder_template
    )

    all_log_like.append(log_like)
    all_wRMSEs.append(wRMSEs)
    all_strengths.append(strengths)

all_log_like = np.array(all_log_like)   # shape: (n_seeds, num_augmentations)
all_wRMSEs = np.array(all_wRMSEs)       # shape: (n_seeds, num_augmentations)

print("\nall_log_like shape:", all_log_like.shape)
print("all_wRMSEs shape:", all_wRMSEs.shape)

# ==================================================
# average and spread across seeds
# ==================================================
mean_log_like = np.nanmean(all_log_like, axis=0)
std_log_like = np.nanstd(all_log_like, axis=0)

mean_wRMSEs = np.nanmean(all_wRMSEs, axis=0)
std_wRMSEs = np.nanstd(all_wRMSEs, axis=0)

# optional: standard error instead of std
n_valid_log = np.sum(~np.isnan(all_log_like), axis=0)
n_valid_wrmse = np.sum(~np.isnan(all_wRMSEs), axis=0)

sem_log_like = std_log_like / np.sqrt(np.maximum(n_valid_log, 1))
sem_wRMSEs = std_wRMSEs / np.sqrt(np.maximum(n_valid_wrmse, 1))

# ==================================================
# colors
# ==================================================
n_colors = 8
cmap = plt.get_cmap("PRGn")
colors = [mpl.colors.to_hex(cmap(i)) for i in np.linspace(0, 1, n_colors)]

# ==================================================
# plot
# ==================================================
augmentations = np.arange(num_augmentations)

fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

# ----------------------------
# negative log likelihood
# ----------------------------
axes[0].plot(
    augmentations,
    -mean_log_like,
    marker="o",
    linewidth=3,
    c=colors[1]
)
axes[0].fill_between(
    augmentations,
    -(mean_log_like + sem_log_like),
    -(mean_log_like - sem_log_like),
    alpha=0.25,
    color=colors[1]
)

axes[0].set_xlabel("augmentation")
axes[0].set_ylabel("negative log likelihood")
axes[0].set_xticks(augmentations)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].locator_params(axis="y", nbins=4)
axes[0].locator_params(axis="x", nbins=5)

# ----------------------------
# wRMSE
# ----------------------------
axes[1].plot(
    augmentations,
    mean_wRMSEs,
    marker="o",
    linewidth=3,
    c=colors[-2]
)
axes[1].fill_between(
    augmentations,
    mean_wRMSEs - sem_wRMSEs,
    mean_wRMSEs + sem_wRMSEs,
    alpha=0.25,
    color=colors[-2]
)

axes[1].set_xlabel("augmentation")
axes[1].set_ylabel("wRMSE")
axes[1].set_xticks(augmentations)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].locator_params(axis="y", nbins=4)
axes[1].locator_params(axis="x", nbins=5)

plt.tight_layout()

png_path = os.path.join(save_plot_dir, f"{save_name}.png")
pdf_path = os.path.join(save_plot_dir, f"{save_name}.pdf")

plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.savefig(pdf_path, dpi=200, bbox_inches="tight")
plt.show()