import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from training.clusterTraining import CygnoTransportModel, build_dataloader, compute_physical_scalars_from_image

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from training.clusterTraining import CygnoTransportModel, build_dataloader
# Assicurati che compute_physical_scalars_from_image sia importata correttamente
# from training.physics_metrics import compute_physical_scalars_from_image 

@torch.no_grad()
def run_validation_sweep_from_dict(
    model, 
    metadata,
    loader,
    inputfile,            
    target_sim_key,       
    sweep_var='H',        
    device=None
):

    model.eval()

    # -----------------------
    # Estrazione delle chiavi REALI esistenti
    # -----------------------
    z_val, alpha_val, lambda_val = target_sim_key
    data_keys = metadata["keys"]["data_keys"]
    
    # Filtriamo solo le chiavi DATA reali che hanno lo z_val corretto
    valid_keys = [k for k in data_keys if k[0] == z_val]
    if not valid_keys:
        raise ValueError(f"Nessuna chiave DATA trovata per z = {z_val}")

    # Ordiniamo le chiavi in base alla variabile su cui vogliamo fare lo sweep
    # k è una tupla: (z, P, T, H) -> P ha indice 1, T indice 2, H indice 3
    if sweep_var == 'H':
        valid_keys_sorted = sorted(valid_keys, key=lambda x: x[3])
    elif sweep_var == 'P':
        valid_keys_sorted = sorted(valid_keys, key=lambda x: x[1])
    elif sweep_var == 'T':
        valid_keys_sorted = sorted(valid_keys, key=lambda x: x[2])
    else:
        raise ValueError("sweep_var deve essere 'H', 'P' o 'T'")

    # Selezioniamo 3 chiavi REALI esistenti: la prima (Min), la centrale (Med), l'ultima (Max)
    idx_min = 0
    idx_med = len(valid_keys_sorted) // 2
    idx_max = len(valid_keys_sorted) - 1

    chosen_keys = [valid_keys_sorted[idx_min], valid_keys_sorted[idx_med], valid_keys_sorted[idx_max]]
        
    # Strutture dati temporanee indicizzate per condizione (tupla)
    # serve a raccogliere tutti i cluster sparsi nei vari batch
    all_data_scalars_dict = {}
    all_pred_scalars_dict = {}
    all_sim_scalars_dict  = {}
    
    atol = 1e-4

    for ib, batch in enumerate(loader):
        print(f"\tProcessing cluster batch # {ib}...")
        
        sim_images = batch["sim_images"].to(device)   # [B, N_sim, H, W]
        sim_cond = batch["sim_cond"].to(device)       # [B, 3]
        data_images = batch["data_images"].to(device) # [B, N_data, H, W]
        data_cond = batch["data_cond"].to(device)     # [B, 4]

        # Forziamo a float per evitare il crash "Float did not match Long"
        batch_data_z = data_cond[:, 0].detach().cpu().numpy()
        batch_data_P = data_cond[:, 1].detach().cpu().numpy()
        batch_data_T = data_cond[:, 2].detach().cpu().numpy()
        batch_data_H = data_cond[:, 3].detach().cpu().numpy()

        batch_sim_z       = sim_cond[:, 0].detach().cpu().numpy()
        batch_sim_alpha   = sim_cond[:, 1].detach().cpu().numpy()
        batch_sim_lambda  = sim_cond[:, 2].detach().cpu().numpy()

        # Ciclo sui 3 bin di interesse per processare SOLO ciò che serve a questo plot
        for current_key in chosen_keys:
            # current_key è già una tupla reale (z, P, T, H)
            target_P = current_key[1]
            target_T = current_key[2]
            target_H = current_key[3]
            
            row_key = (target_P, target_T, target_H)
            if row_key not in all_data_scalars_dict:
                all_data_scalars_dict[row_key] = []
                all_pred_scalars_dict[row_key] = []
                all_sim_scalars_dict[row_key]  = []

            data_mask = (
                (np.abs(batch_data_z - z_val) < atol) & 
                (np.abs(batch_data_P - target_P) < atol) & 
                (np.abs(batch_data_T - target_T) < atol) & 
                (np.abs(batch_data_H - target_H) < atol)
            )

            if np.any(data_mask):
                # data_mask è un vettore booleano NumPy 1D di lunghezza B. 
                # Lo usiamo per indicizzare il tensore PyTorch data_images
                matched_data = data_images[torch.from_numpy(data_mask).to(data_images.device)]
                B_d, N_d, H_d, W_d = matched_data.shape
                data_flat = matched_data.view(B_d * N_d, 1, H_d, W_d)
                data_scalars = compute_physical_scalars_from_image(data_flat)
                all_data_scalars_dict[row_key].append(data_scalars.cpu().numpy())

            # --- MASCHERA E GENERAZIONE MODELLO (SIM -> PRED) ---
            sim_mask = (
                (np.abs(batch_sim_z - z_val) < atol) & 
                (np.abs(batch_sim_alpha - alpha_val) < atol) & 
                (np.abs(batch_sim_lambda - lambda_val) < atol)
            )
            
            if sim_mask.any():
                torch_sim_mask = torch.from_numpy(sim_mask).to(sim_images.device)
                matched_sim = sim_images[torch_sim_mask]
                matched_cond = sim_cond[torch_sim_mask]
                
                B_s, N_s, H_s, W_s = matched_sim.shape
                sim_flat = matched_sim.view(B_s * N_s, 1, H_s, W_s)
                
                sim_cond_flat = matched_cond[:, None, :].repeat(1, N_s, 1).view(B_s * N_s, 3)
                target_cond_tensor = torch.tensor([[z_val, target_P, target_T, target_H]], dtype=torch.float32).to(device)
                target_cond_flat = target_cond_tensor.repeat(B_s * N_s, 1)
                
                sim_scalars_flat = compute_physical_scalars_from_image(sim_flat)
                
                out = model(sim_flat, sim_cond_flat, target_cond_flat, sim_scalars_flat)
                pred_clamped_flat = F.elu(out["pred_images"]) + 1.0
                pred_scalars = compute_physical_scalars_from_image(pred_clamped_flat)
                
                all_pred_scalars_dict[row_key].append(pred_scalars.cpu().numpy())
                all_sim_scalars_dict[row_key].append(sim_scalars_flat.cpu().numpy())

    # -----------------------
    # 4. Configurazione e generazione dei Grafici
    # -----------------------
    print("===> Done filling distributions. Now plotting...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"Variable {sweep_var} | Central SIM: z={z_val} cm, $\\alpha$={alpha_val}, $\lambda$={lambda_val} mm", fontsize=16)
    scalar_names = ['Integral (counts)', 'Length (pix)', 'Width (pix)']
    
    bins = {}
    for row_idx, current_key in enumerate(chosen_keys):
        target_P = current_key[1]
        target_T = current_key[2]
        target_H = current_key[3]
        
        # Valore della variabile corrente per l'etichetta
        if sweep_var == 'H': current_val = target_H
        elif sweep_var == 'P': current_val = target_P
        elif sweep_var == 'T': current_val = target_T
        
        row_key = (target_P, target_T, target_H)
        
        # Concatenazione finale per questa specifica riga dello sweep
        real_scalars = np.concatenate(all_data_scalars_dict[row_key], axis=0) if all_data_scalars_dict[row_key] else np.empty((0, 3))
        pred_scalars = np.concatenate(all_pred_scalars_dict[row_key], axis=0) if all_pred_scalars_dict[row_key] else np.empty((0, 3))
        sim_scalars  = np.concatenate(all_sim_scalars_dict[row_key], axis=0)  if all_sim_scalars_dict[row_key] else np.empty((0, 3))

        for col_idx, scalar_name in enumerate(scalar_names):
            ax = axes[row_idx, col_idx]

            print (f"Plot (row_{current_val},col_{scalar_name}) = {row_idx,col_idx}")
            print (f"\tThe datasets have entries (sim,pred,data) = ({len(sim_scalars)},{len(pred_scalars)},{len(real_scalars)})")

            # Range dinamico
            all_vals = np.concatenate([sim_scalars[:, col_idx], pred_scalars[:, col_idx], real_scalars[:, col_idx]])
            vmin, vmax = np.percentile(all_vals, 1.0), np.percentile(all_vals, 99.0)
            if row_idx==0:
                bins[col_idx] = np.linspace(vmin, vmax, 30)
            
            # A. SIM DI PARTENZA (Riempimento Blu semi-trasparente)
            if len(sim_scalars) > 0:
                ax.hist(sim_scalars[:, col_idx], bins=bins[col_idx], alpha=0.5, histtype="step", linewidth=2, density=True, label='Central SIM', color='tab:blue')
                        
            # B. PRED TRASPORTATI (Riempimento Rosso/Arancio semi-trasparente)
            if len(pred_scalars) > 0:
                ax.hist(pred_scalars[:, col_idx], bins=bins[col_idx], alpha=0.7, histtype="step", linewidth=2, density=True, label='Corr. SIM', color='tab:orange')

            # B. DATA TARGET REALE (Linea verde spessa, svuotato dentro per vedere lo sfondo)
            if len(real_scalars) > 0:
                # Plot Dati con Errore
                counts, bin_edges = np.histogram(real_scalars[:, col_idx], bins=bins[col_idx])
                counts_density, _ = np.histogram(real_scalars[:, col_idx], bins=bins[col_idx], density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            
                # Calcolo errore poissoniano densità
                valid_bins = counts > 0
                scaling_factor = np.divide(
                    counts_density, 
                    counts, 
                    out=np.zeros_like(counts_density), 
                    where=(counts > 0)
                )
                errors_density = np.sqrt(counts) * scaling_factor
            
                ax.errorbar(bin_centers[valid_bins], counts_density[valid_bins], yerr=errors_density[valid_bins], 
                            fmt='o', markersize=4, color="black", capsize=2, label="Data")

             # Formattazione estetica dei subplot
            if row_idx == 0:
                ax.set_title(scalar_name, fontweight='bold', fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{sweep_var} = {current_val}\nP={target_P}bar, T={target_T}C, H={target_H}?\n\nClusters", rotation=90, labelpad=10, fontsize=10)
            
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_dir = "plot/validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    sim_key_str = "_".join(f"{x:g}".replace(".", "p") for x in target_sim_key)
    for ext in ["png","pdf"]:
        plotname = os.path.join(output_dir, f"fullstat_validation_variable_{sweep_var}_SIM_{sim_key_str}.{ext}")
        plt.savefig(plotname, dpi=120)
        print(f"Saved validation plot in {plotname}")
    plt.close()


