#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:26:45 2026

@author: dimi
"""

import os
import re
import glob
import pickle
import numpy as np

Ts = [200, 300, 400, 500]
obs_denz = [160, 200, 240, 280]
seeds = [11, 12, 13]
strengths =  [ 0.1 , 0.5, 1.0, 2.0]
dsystem = "LC"
g = 0.50

base_root = "/Volumes/KINGSTON/geometric_more"

tmp = {}

metric_names = [
    "w2_avg_aug0",
    "cov_diff_avg_aug0",
    "mean_diff_avg_aug0",
    "wRMSE_1_aug0",
    "wRMSE_1_aug1",
]

for T in Ts:
    for obs_den in obs_denz:
        for seed in seeds:

            folder = (
                f"{base_root}/VanDerGeodesic_augmentation_"
                f"{dsystem}_noise_{g:.2f}_obs_dens_{obs_den}_sim_time_{T}_seed_{seed}/"
            )

            if not os.path.isdir(folder):
                continue

            # -----------------------------
            # augmentation 0
            # -----------------------------
            pattern0 = os.path.join(folder, "f_est_after_0_augm_n_eval_beta_*.dat")
            files0 = sorted(glob.glob(pattern0))

            for fpath in files0:
                fname = os.path.basename(fpath)
                m = re.search(r"beta_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.dat$", fname)
                if m is None:
                    continue
                beta = float(m.group(1))

                try:
                    with open(fpath, "rb") as f:
                        data = pickle.load(f)
                except Exception:
                    continue

                if beta not in tmp:
                    tmp[beta] = {}
                for metric in metric_names:
                    if metric not in tmp[beta]:
                        tmp[beta][metric] = {}
                    if (obs_den, T) not in tmp[beta][metric]:
                        tmp[beta][metric][(obs_den, T)] = []

                brdist = data.get("brdistances", {})

                w2_list = []
                cov_list = []
                mean_list = []

                # aggregate over all numeric keys inside brdistances
                for k, subd in brdist.items():
                    is_numeric_key = False

                    if isinstance(k, (int, float, np.integer, np.floating)):
                        is_numeric_key = True
                    elif isinstance(k, str):
                        try:
                            float(k)
                            is_numeric_key = True
                        except Exception:
                            is_numeric_key = False

                    if not is_numeric_key:
                        continue

                    if not isinstance(subd, dict):
                        continue

                    val = subd.get("w2_avg", np.nan)
                    try:
                        w2_list.append(float(val))
                    except Exception:
                        pass

                    val = subd.get("cov_diff_avg", np.nan)
                    try:
                        cov_list.append(float(val))
                    except Exception:
                        pass

                    val = subd.get("mean_diff_avg", np.nan)
                    try:
                        mean_list.append(float(val))
                    except Exception:
                        pass

                # mean over numeric keys within this file
                file_w2_mean = np.nan if len(w2_list) == 0 else np.nanmean(w2_list)
                file_cov_mean = np.nan if len(cov_list) == 0 else np.nanmean(cov_list)
                file_mean_mean = np.nan if len(mean_list) == 0 else np.nanmean(mean_list)

                tmp[beta]["w2_avg_aug0"][(obs_den, T)].append(file_w2_mean)
                tmp[beta]["cov_diff_avg_aug0"][(obs_den, T)].append(file_cov_mean)
                tmp[beta]["mean_diff_avg_aug0"][(obs_den, T)].append(file_mean_mean)

                try:
                    tmp[beta]["wRMSE_1_aug0"][(obs_den, T)].append(float(data.get("wRMSE_1", np.nan)))
                except Exception:
                    tmp[beta]["wRMSE_1_aug0"][(obs_den, T)].append(np.nan)

            # -----------------------------
            # augmentation 1
            # -----------------------------
            pattern1 = os.path.join(folder, "f_est_after_1_augm_n_eval_beta_*.dat")
            files1 = sorted(glob.glob(pattern1))

            for fpath in files1:
                fname = os.path.basename(fpath)
                m = re.search(r"beta_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.dat$", fname)
                if m is None:
                    continue
                beta = float(m.group(1))

                try:
                    with open(fpath, "rb") as f:
                        data = pickle.load(f)
                except Exception:
                    continue

                if beta not in tmp:
                    tmp[beta] = {}
                if "wRMSE_1_aug1" not in tmp[beta]:
                    tmp[beta]["wRMSE_1_aug1"] = {}
                if (obs_den, T) not in tmp[beta]["wRMSE_1_aug1"]:
                    tmp[beta]["wRMSE_1_aug1"][(obs_den, T)] = []

                try:
                    tmp[beta]["wRMSE_1_aug1"][(obs_den, T)].append(float(data.get("wRMSE_1", np.nan)))
                except Exception:
                    tmp[beta]["wRMSE_1_aug1"][(obs_den, T)].append(np.nan)
                    
#%%                    


#%%

results = {}

for beta in sorted(tmp.keys()):
    results[beta] = {}

    for metric in metric_names:
        grid = np.full((len(obs_denz), len(Ts)), np.nan, dtype=float)

        for i_obs, obs_den in enumerate(obs_denz):
            for i_T, T in enumerate(Ts):
                vals = tmp[beta].get(metric, {}).get((obs_den, T), [])
                vals = np.asarray(vals, dtype=float)

                if vals.size == 0 or np.all(np.isnan(vals)):
                    grid[i_obs, i_T] = np.nan
                else:
                    grid[i_obs, i_T] = np.nanmean(vals)

        results[beta][metric] = grid
        
        
#%%



import matplotlib.pyplot as plt
import numpy as np
import os

# results[beta][metric] is assumed to exist from the previous code
# with grids of shape (len(obs_denz), len(Ts))

for beta in sorted(results.keys()):

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    metric_list = [
        ("w2_avg_aug0", "augmentation 0\nmean w2_avg"),
        ("cov_diff_avg_aug0", "augmentation 0\nmean cov_diff_avg"),
        ("mean_diff_avg_aug0", "augmentation 0\nmean mean_diff_avg"),
        ("wRMSE_1_aug0", "augmentation 0\nwRMSE_1"),
        ("wRMSE_1_aug1", "augmentation 1\nwRMSE_1"),
    ]

    for ax, (metric, title) in zip(axes[:5], metric_list):
        grid = results[beta][metric]

        im = ax.imshow(grid, origin="lower", aspect="auto")
        ax.set_title(title)

        ax.set_xticks(np.arange(len(Ts)))
        ax.set_xticklabels(Ts)
        ax.set_yticks(np.arange(len(obs_denz)))
        ax.set_yticklabels(obs_denz)

        ax.set_xlabel("T")
        ax.set_ylabel("obs_dens")

        # annotate each cell
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                txt = "nan" if np.isnan(val) else f"{val:.3g}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[5].axis("off")

    fig.suptitle(f"beta = {beta:.2f}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
#%%

save_plot_dir = "/path/to/save/plots"
os.makedirs(save_plot_dir, exist_ok=True)

for beta in sorted(results.keys()):

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    metric_list = [
        ("w2_avg_aug0", "augmentation 0\nmean w2_avg"),
        ("cov_diff_avg_aug0", "augmentation 0\nmean cov_diff_avg"),
        ("mean_diff_avg_aug0", "augmentation 0\nmean mean_diff_avg"),
        ("wRMSE_1_aug0", "augmentation 0\nwRMSE_1"),
        ("wRMSE_1_aug1", "augmentation 1\nwRMSE_1"),
    ]

    for ax, (metric, title) in zip(axes[:5], metric_list):
        grid = results[beta][metric]

        im = ax.imshow(grid, origin="lower", aspect="auto")
        ax.set_title(title)

        ax.set_xticks(np.arange(len(Ts)))
        ax.set_xticklabels(Ts)
        ax.set_yticks(np.arange(len(obs_denz)))
        ax.set_yticklabels(obs_denz)

        ax.set_xlabel("T")
        ax.set_ylabel("obs_dens")

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                txt = "nan" if np.isnan(val) else f"{val:.3g}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[5].axis("off")

    fig.suptitle(f"beta = {beta:.2f}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_plot_dir, f"metrics_beta_{beta:.2f}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
    #%%
    
    
import numpy as np
import matplotlib.pyplot as plt

# assumes:
# results[beta]["wRMSE_1_aug0"] and results[beta]["wRMSE_1_aug1"] already exist
# each grid has shape (len(obs_denz), len(Ts))

betas_to_plot = [0.1, 0.5, 1.0, 2.0]

import numpy as np
import matplotlib.pyplot as plt

betas_to_plot = [0.1, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = np.asarray(axes)

# collect all values to set one common color scale
all_vals = []
for beta in betas_to_plot:
    if beta in results:
        all_vals.append(results[beta]["wRMSE_1_aug0"])
        all_vals.append(results[beta]["wRMSE_1_aug1"])

stacked = np.stack(all_vals)
vmin = np.nanmin(stacked)
vmax = np.nanmax(stacked)

last_im = None

# first row: augmentation 0
for j, beta in enumerate(betas_to_plot):
    ax = axes[0, j]

    if beta in results:
        grid = results[beta]["wRMSE_1_aug0"]
    else:
        grid = np.full((len(obs_denz), len(Ts)), np.nan)

    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    last_im = im

    ax.set_title(f"beta = {beta:g}, augmentation 0")
    ax.set_xticks(np.arange(len(Ts)))
    ax.set_xticklabels(Ts)
    ax.set_yticks(np.arange(len(obs_denz)))
    ax.set_yticklabels(obs_denz)
    ax.set_xlabel("T")
    ax.set_ylabel("obs_dens")

    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            val = grid[i, k]
            txt = "nan" if np.isnan(val) else f"{val:.3g}"
            ax.text(k, i, txt, ha="center", va="center", fontsize=9, color="white")

# second row: augmentation 1
for j, beta in enumerate(betas_to_plot):
    ax = axes[1, j]

    if beta in results:
        grid = results[beta]["wRMSE_1_aug1"]
    else:
        grid = np.full((len(obs_denz), len(Ts)), np.nan)

    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    last_im = im

    ax.set_title(f"beta = {beta:g}, augmentation 1")
    ax.set_xticks(np.arange(len(Ts)))
    ax.set_xticklabels(Ts)
    ax.set_yticks(np.arange(len(obs_denz)))
    ax.set_yticklabels(obs_denz)
    ax.set_xlabel("T")
    ax.set_ylabel("obs_dens")

    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            val = grid[i, k]
            txt = "nan" if np.isnan(val) else f"{val:.3g}"
            ax.text(k, i, txt, ha="center", va="center", fontsize=9, color="white")

# one shared colorbar for all subplots
cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label("wRMSE_1")

plt.tight_layout()
plt.show()


#%%



import numpy as np
import matplotlib.pyplot as plt

from plotting_config import update_rc_params
update_rc_params()

betas_to_plot = [0.1, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(2, 4, figsize=(10, 4.5))
axes = np.asarray(axes)

# common color scale
all_vals = []
for beta in betas_to_plot:
    if beta in results:
        all_vals.append(results[beta]["wRMSE_1_aug0"])
        all_vals.append(results[beta]["wRMSE_1_aug1"])

stacked = np.stack(all_vals)
vmin = np.nanmin(stacked)
vmax = np.nanmax(stacked)

last_im = None

# first row: augmentation 0
for j, beta in enumerate(betas_to_plot):
    ax = axes[0, j]

    if beta in results:
        grid = results[beta]["wRMSE_1_aug0"]
    else:
        grid = np.full((len(obs_denz), len(Ts)), np.nan)

    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                   cmap="magma_r")
    last_im = im

    #ax.set_title(f"beta = {beta:g}, augmentation 0")
    ax.set_xticks(np.arange(len(Ts)))
    ax.set_xticklabels(Ts,fontsize=12)
    ax.set_yticks(np.arange(len(obs_denz)))
    ax.set_yticklabels(obs_denz,fontsize=12)
    ax.set_xlabel("T",fontsize=16)
    ax.set_ylabel(r"$\tau$",fontsize=16)
    #ax.set_rasterized(True)

    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            val = grid[i, k]
            txt = "nan" if np.isnan(val) else f"{val:.2g}"
            ax.text(k, i, txt, ha="center", va="center", fontsize=9, color="white")

# second row: augmentation 1
for j, beta in enumerate(betas_to_plot):
    ax = axes[1, j]

    if beta in results:
        grid = results[beta]["wRMSE_1_aug1"]
    else:
        grid = np.full((len(obs_denz), len(Ts)), np.nan)

    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                   cmap="magma_r",  interpolation='None')
    last_im = im

    #ax.set_title(f"beta = {beta:g}, augmentation 1")
    ax.set_xticks(np.arange(len(Ts)))
    ax.set_xticklabels(Ts,fontsize=12)
    ax.set_yticks(np.arange(len(obs_denz)))
    ax.set_yticklabels(obs_denz,fontsize=12)
    ax.set_xlabel("T",fontsize=16)
    ax.set_ylabel(r"$\tau$",fontsize=16)
    #ax.set_rasterized(True)

    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            val = grid[i, k]
            txt = "nan" if np.isnan(val) else f"{val:.2g}"
            ax.text(k, i, txt, ha="center", va="center", fontsize=9, color="white")

# leave space at bottom for horizontal colorbar
fig.tight_layout(rect=[0, 0.12, 1, 1])
cax = fig.add_axes([0.2, -0.015, 0.6, 0.035])  # [left, bottom, width, height]
cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
#cbar.set_label("wRMSE_1")
cbar.set_label("wRMSE", fontsize=16)
plt.savefig('Sensitivity_for_beta2.png', dpi=200)
plt.savefig('Sensitivity_for_beta2.pdf')
plt.savefig('Sensitivity_for_beta2.svg', dpi=200)
plt.show()