import torch
import os,random
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
mplhep.style.use([mplhep.style.CMS])
import mplhep as hep

from scipy.stats import wasserstein_distance
from matplotlib import colors
from plot.plot_utils import var_titles

def compute_validation_metrics(A_corr, A_data):
    """
    A_corr, A_data: tensor (N, 2)
    """

    A_corr_np = A_corr.detach().cpu().numpy()
    A_data_np = A_data.detach().cpu().numpy()

    metrics = {}

    # --- 1D Wasserstein (per ciascuna variabile)
    w1_0 = wasserstein_distance(A_corr_np[:, 0], A_data_np[:, 0])
    w1_1 = wasserstein_distance(A_corr_np[:, 1], A_data_np[:, 1])
    metrics["w1_mean"] = 0.5 * (w1_0 + w1_1)

    # --- correlazione
    rho_corr = np.corrcoef(A_corr_np.T)[0, 1]
    rho_data = np.corrcoef(A_data_np.T)[0, 1]
    metrics["delta_rho"] = abs(rho_corr - rho_data)

    # --- covarianza
    cov_corr = np.cov(A_corr_np.T)
    cov_data = np.cov(A_data_np.T)
    metrics["cov_fro"] = np.linalg.norm(cov_corr - cov_data, ord="fro")

    # --- MMD 2D (RBF kernel semplice)
    def rbf_kernel(x, y, sigma=1.0):
        x = x[:, None, :]
        y = y[None, :, :]
        return np.exp(-np.sum((x - y)**2, axis=-1) / (2 * sigma**2))

    Kxx = rbf_kernel(A_corr_np, A_corr_np)
    Kyy = rbf_kernel(A_data_np, A_data_np)
    Kxy = rbf_kernel(A_corr_np, A_data_np)

    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    metrics["mmd_2d"] = mmd

    return metrics

def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()
    agg = {}

    for k in keys:
        agg[k] = np.mean([m[k] for m in metrics_list])

    return agg



def plot_2d_comparison(A_sim, A_corr, A_data, variables_to_plot, path, params=None, suffix=None ):
    if len(variables_to_plot)>2:
        print("ERROR! Can plot the correlation of 2 variables only.")
        return

    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    A_sim = A_sim.detach().cpu().numpy()
    A_corr = A_corr.detach().cpu().numpy()
    A_data = A_data.detach().cpu().numpy()

    datasets = [A_sim, A_data, A_corr]
    titles   = ["Sim", "Data", "Sim (corr)"]

    # --- 1. range globale con percentili ---
    all_data = np.vstack(datasets)
    xmin, xmax = np.percentile(all_data[:,0], [1, 99])
    ymin, ymax = np.percentile(all_data[:,1], [1, 99])

    # --- 2. calcolo istogrammi con stesso range ---
    histos = []
    edges  = []
    for data in datasets:
        H, xedges, yedges = np.histogram2d(
            data[:,0], data[:,1],
            bins=50,
            range=[[xmin, xmax], [ymin, ymax]]
        )
        histos.append(H)
        edges.append((xedges, yedges))

    # --- 3. plot ---
    for ax, data, title in zip(axs, datasets, titles):
        H, xedges, yedges = np.histogram2d(
        data[:,0], data[:,1],
        bins=30,
        range=[[xmin, xmax], [ymin, ymax]]
        )
        # Sopprime zeri impostando min > 0 (evita colori falsi)
        H[H == 0] = np.nan
        hep.hist2dplot(H, xedges, yedges, ax=ax, cmap="rainbow", cbar=False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.set_xlabel(var_titles[variables_to_plot[0]])
        ax.set_ylabel(var_titles[variables_to_plot[1]])
        
    # Remove the space between the subplots
    plt.subplots_adjust(hspace=0)
    axs[0].margins(x=0)
    axs[1].margins(x=0)
    axs[2].margins(x=0)

    # Adjust the tight_layout to not add extra padding
    fig.tight_layout()
    #fig.subplots_adjust(left=0.65, right=0.95, wspace=0.2)
    
    # fig.text(0., 0.80, r'$^{55}Fe$ data/sim', fontsize=20, verticalalignment='top')
    # if params:
    #     text = f"z = {params['ztrue_val']} cm\nSim: $\lambda_{{abs}}$={params['lambda_val']}mm, $\\alpha$={params['alpha_val']}\nData: P={params['P_val']}bar, T={params['T_val']}C"
    #     fig.text(0.0,0.5, text, fontsize=20, verticalalignment='center')

        
    # Plot and save the histograms
    suff=f"_{suffix}" if suffix else ""
    suff = suff.replace(".","p")
    output_path = os.path.join(path, f"{variables_to_plot[1]}_vs_{variables_to_plot[0]}{suff}")

    print(f"===> 2D Validation save plot {output_path}.pdf/png")
    for ext in ["png","pdf"]:
        fig.savefig(f"{output_path}.{ext}")
    plt.close()

def random_ordered_pair(lst):
    i = random.randint(0, len(lst) - 2)
    j = random.randint(i + 1, len(lst) - 1)
    return [lst[i], lst[j]]
