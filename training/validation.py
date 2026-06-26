import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from training.clusterTraining import CygnoTransportModel, build_dataloader
# Assicurati che compute_physical_scalars_from_image sia importata correttamente dal tuo modulo fisco
# from training.physics_metrics import compute_physical_scalars_from_image 

@torch.no_grad()
def run_validation_sweep_from_dict(
    model_or_path, 
    metadata,             # Il tuo dizionario caricato dal pickle
    inputfile,            # Aggiunto per permettere a build_dataloader di funzionare
    target_sim_key,       # Tupla di partenza, es: (10.0, 0.5, 400)
    sweep_var='H',        # Variabile da iterare: 'H', 'P' o 'T'
    device=None
):

    # -----------------------
    # device setup
    # -----------------------
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # -----------------------
    # load model if needed
    # -----------------------
    if isinstance(model_or_path, str):
        print(f"\nLoading model from:\n{model_or_path}")
        model = CygnoTransportModel().to(device)
        state = torch.load(model_or_path, map_location=device)
        model.load_state_dict(state)
    else:
        model = model_or_path.to(device)
        
    model.eval()

    # Costruiamo il dataloader usando il file di input passato
    _, loader = build_dataloader(inputfile, batch_size=64, is_test=True)
    
    # 1. Parsing della chiave SIM
    z_val, alpha_val, lambda_val = target_sim_key
    
    # 2. Estrazione e calcolo dei valori intermedi dai metadati
    data_keys = metadata["keys"]["data_keys"]
    
    # Filtriamo solo le chiavi DATA che corrispondono allo z_val della SIM
    valid_keys = [k for k in data_keys if k[0] == z_val]
    if not valid_keys:
        raise ValueError(f"Nessuna chiave DATA trovata per z = {z_val}")

    # Estraiamo i valori unici ordinati per P (idx 1), T (idx 2), H (idx 3)
    unique_P = sorted(list(set([k[1] for k in valid_keys])))
    unique_T = sorted(list(set([k[2] for k in valid_keys])))
    unique_H = sorted(list(set([k[3] for k in valid_keys])))
    
    # Troviamo i valori mediani
    med_P = unique_P[len(unique_P)//2]
    med_T = unique_T[len(unique_T)//2]
    med_H = unique_H[len(unique_H)//2]
    
    # 3. Impostazione della logica di Sweep (3 Bin)
    if sweep_var == 'H':
        sweep_values = [unique_H[0], med_H, unique_H[-1]]
        fixed_conds = {'P': med_P, 'T': med_T}
    elif sweep_var == 'P':
        sweep_values = [unique_P[0], med_P, unique_P[-1]]
        fixed_conds = {'T': med_T, 'H': med_H}
    elif sweep_var == 'T':
        sweep_values = [unique_T[0], med_T, unique_T[-1]]
        fixed_conds = {'P': med_P, 'H': med_H}
    else:
        raise ValueError("sweep_var deve essere 'H', 'P' o 'T'")
    
    # Setup della Figura principale (3 righe per i bin dello sweep, 3 colonne per le metriche)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"Sweep su {sweep_var} | SIM Base: z={z_val}, alpha={alpha_val}, lambda={lambda_val}", fontsize=16)
    scalar_names = ['Integral', 'Length', 'Width']
    
    # 5. Loop sulle 3 righe (i valori della variabile di sweep)
    for row_idx, current_val in enumerate(sweep_values):
        
        # Costruisci la chiave DATA target per questa riga
        target_P = current_val if sweep_var == 'P' else fixed_conds['P']
        target_T = current_val if sweep_var == 'T' else fixed_conds['T']
        target_H = current_val if sweep_var == 'H' else fixed_conds['H']
        
        # Tensore condizione target per il modello: [1, 4] -> [z, P, T, H]
        target_cond_tensor = torch.tensor([[z_val, target_P, target_T, target_H]], dtype=torch.float32).to(device)
        
        all_pred_scalars = []
        all_sim_scalars = []
        all_data_scalars = []
        
        # INFERENCE & EXTRACTION LOOP sul dataloader generico
        for i, batch in enumerate(loader):
            sim_images = batch["sim_images"].to(device)  # [B, N_sim, H, W]
            sim_cond = batch["sim_cond"].to(device)      # [B, 3]
            data_images = batch["data_images"].to(device) # [B, N_data, H, W]
            data_cond = batch["data_cond"].to(device)    # [B, 4]

            # Estrazione colonne per maschere logiche
            batch_data_z, batch_data_P, batch_data_T, batch_data_H = data_cond[:, 0], data_cond[:, 1], data_cond[:, 2], data_cond[:, 3]
            batch_sim_z, batch_sim_alpha, batch_sim_lambda = sim_cond[:, 0], sim_cond[:, 1], sim_cond[:, 2]

            atol = 1e-4 
            
            # Maschera DATA reale per questa riga di sweep
            data_mask = (
                torch.isclose(batch_data_z, torch.tensor(z_val).to(device), atol=atol) & 
                torch.isclose(batch_data_P, torch.tensor(target_P).to(device), atol=atol) & 
                torch.isclose(batch_data_T, torch.tensor(target_T).to(device), atol=atol) & 
                torch.isclose(batch_data_H, torch.tensor(target_H).to(device), atol=atol)
            )
            
            # Maschera SIM sorgente di partenza
            sim_mask = (
                torch.isclose(batch_sim_z, torch.tensor(z_val).to(device), atol=atol) & 
                torch.isclose(batch_sim_alpha, torch.tensor(alpha_val).to(device), atol=atol) & 
                torch.isclose(batch_sim_lambda, torch.tensor(lambda_val).to(device), atol=atol)
            )
            
            # --- PARTE 1: Estrazione Scalari DATA Reali ---
            if data_mask.any():
                matched_data = data_images[data_mask] # [Count_data, N_data, H, W]
                B_d, N_d, H_d, W_d = matched_data.shape
                data_flat = matched_data.view(B_d * N_d, 1, H_d, W_d)
                
                data_scalars = compute_physical_scalars_from_image(data_flat)
                all_data_scalars.append(data_scalars.cpu().numpy())

            # --- PARTE 2: Modello ed Estrazione Predizioni (SIM -> PRED) ---
            if sim_mask.any():
                matched_sim = sim_images[sim_mask]     # [Count_sim, N_sim, H, W]
                matched_cond = sim_cond[sim_mask]      # [Count_sim, 3]
                
                B_s, N_s, H_s, W_s = matched_sim.shape
                sim_flat = matched_sim.view(B_s * N_s, 1, H_s, W_s)
                
                # Spatialize/Repeat delle condizioni per matchare il totale dei cluster (B_s * N_s)
                sim_cond_flat = matched_cond[:, None, :].repeat(1, N_s, 1).view(B_s * N_s, 3)
                target_cond_flat = target_cond_tensor.repeat(B_s * N_s, 1)
                
                # Calcolo scalari della SIM di input
                sim_scalars_flat = compute_physical_scalars_from_image(sim_flat)
                
                # Forward Pass del modello di Trasporto
                out = model(sim_flat, sim_cond_flat, target_cond_flat, sim_scalars_flat)
                pred_clamped_flat = F.elu(out["pred_images"]) + 1.0
                
                # Calcolo scalari sulle immagini trasportate (PRED)
                pred_scalars = compute_physical_scalars_from_image(pred_clamped_flat)
                
                all_pred_scalars.append(pred_scalars.cpu().numpy())
                all_sim_scalars.append(sim_scalars_flat.cpu().numpy())
            
        # Concatena i vettori accumulati da tutti i batch per la riga corrente
        real_scalars = np.concatenate(all_data_scalars, axis=0) if all_data_scalars else np.empty((0, 3))
        pred_scalars = np.concatenate(all_pred_scalars, axis=0) if all_pred_scalars else np.empty((0, 3))
        sim_scalars  = np.concatenate(all_sim_scalars, axis=0)  if all_sim_scalars else np.empty((0, 3))

        # 6. Plotting delle 3 colonne (Metriche 1D)
        for col_idx, scalar_name in enumerate(scalar_names):
            ax = axes[row_idx, col_idx]
            
            # SIM Base (Blu)
            if len(sim_scalars) > 0:
                ax.hist(sim_scalars[:, col_idx], bins=50, alpha=0.4, density=True, label='SIM Base', color='blue')
            
            # DATA Reali Target (Verde - Istogramma vuoto a step con linea spessa)
            if len(real_scalars) > 0:
                ax.hist(real_scalars[:, col_idx], bins=50, density=True, label='DATA Target', color='green', histtype='step', linewidth=2)
            
            # PRED Trasportati dal modello (Rosso)
            if len(pred_scalars) > 0:
                ax.hist(pred_scalars[:, col_idx], bins=50, alpha=0.4, density=True, label='PRED Trasportati', color='red')
            
            # Formattazione estetica dei subplot
            if row_idx == 0:
                ax.set_title(scalar_name, fontweight='bold', fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{sweep_var} = {current_val}\nP={target_P}, T={target_T}, H={target_H}\n\nDensità", rotation=90, labelpad=10, fontsize=10)
            
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
