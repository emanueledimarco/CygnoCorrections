# python libraries import
import os 
import numpy as np
import glob
import scipy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, context_dim=0, mask=None, hidden_dim=128):
        """
        dim         : numero totale di feature x
        context_dim : dimensione del contesto condizionale
        mask        : tensor (dim,), 1 = mascherata (invariate), 0 = da trasformare
        hidden_dim  : dimensione della rete interna s,t
        """
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim

        if mask is None:
            mask = torch.zeros(dim)
            mask[::2] = 1  # alterna 1 e 0
        self.register_buffer("mask", mask)

        self.dim_masked = int(mask.sum().item())
        self.dim_unmasked = dim - self.dim_masked

        # rete affine s,t
        input_dim = self.dim_masked + context_dim
        self.st_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.dim_unmasked)
        )

    def forward(self, x, context=None):
        """
        x: (B, dim) o (B, N, dim)
        context: (B, context_dim) o (B*N, context_dim)
        """
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])  # appiattisce batch ed eventuali dimensioni extra

        base_mask = self.mask.to(x.device)
        base_mask_bool = base_mask.bool()
        base_dim = base_mask_bool.numel()

        # --- x_masked per st_net: SOLO feature osservabili ---
        x_obs = x_flat[:, :base_dim]  
        x_masked = x_obs[:, base_mask_bool]

        # --- mask esteso SOLO per decidere cosa trasformare ---
        if x_flat.shape[1] > base_mask.shape[0]:
            n_latent = x_flat.shape[1] - base_mask.shape[0]
            latent_mask = torch.ones(n_latent, device=x.device)
            full_mask = torch.cat([base_mask, latent_mask], dim=0)
        else:
            full_mask = base_mask

        full_mask_bool = full_mask.bool()
        
        # concatena context
        # context: (C,) o (1, C)
        if context is not None:
            if context.dim() == 1:
                # da (C,) a (N_events, C)
                context_flat = context.unsqueeze(0).expand(x_masked.shape[0], -1)
            else:
                # già (1, C)
                context_flat = context.expand(x_masked.shape[0], -1)
            net_input = torch.cat([x_masked, context_flat], dim=1)
        else:
            net_input = x_masked

        # calcola scale e shift
        s, t = self.st_net(net_input).chunk(2, dim=1)
        s = torch.tanh(s)  # stabilizza numericamente

        # applica affine solo alle feature non mascherate
        y_flat = x_flat.clone()
        y_flat[:, (~full_mask_bool)] = x_flat[:, (~full_mask_bool)] * torch.exp(s) + t
        
        # rimetti la forma originale
        y = y_flat.view(*orig_shape)
        log_det = s.sum(dim=1)  # shape (B*N,) se batch flatten, puoi rimodellare se vuoi

        return y, log_det

    def inverse(self, y, context=None):
        """
        y: (B, dim) o (B, N, dim)
        context: (B, context_dim) o (B*N, context_dim)
        """
        orig_shape = y.shape
        y_flat = y.view(-1, y.shape[-1])

        base_mask = self.mask.to(y.device)
        base_mask_bool = base_mask.bool()
        base_dim = base_mask_bool.numel()
        
        # --- x_masked per st_net: SOLO feature osservabili ---
        y_obs = y_flat[:, :base_dim]
        y_masked = y_obs[:, base_mask_bool]

        # --- mask esteso SOLO per decidere cosa trasformare ---
        if y_flat.shape[1] > base_mask.shape[0]:
            n_latent = y_flat.shape[1] - base_mask.shape[0]
            latent_mask = torch.ones(n_latent, device=y.device)
            full_mask = torch.cat([base_mask, latent_mask], dim=0)
        else:
            full_mask = base_mask

        full_mask_bool = full_mask.bool()


        
        if context is not None:
            context_flat = context.view(y_masked.shape[0], -1)
            net_input = torch.cat([y_masked, context_flat], dim=1)
        else:
            net_input = y_masked

        s, t = self.st_net(net_input).chunk(2, dim=1)
        s = torch.tanh(s)

        x_flat = y_flat.clone()
        x_flat[:, (~full_mask_bool)] = (y_flat[:, (~full_mask_bool)] - t) * torch.exp(-s)
        
        x = x_flat.view(*orig_shape)
        log_det = -s.sum(dim=1)

        return x    

def sanity_check_coupling(flow, context_encoder, device="cpu"):
    flow.eval()
    context_encoder.eval()

    with torch.no_grad():
        B = 10
        x = torch.randn(B, flow.dim)
        z = torch.randn_like(x)
        x_in = torch.cat([x, z], dim=1)
        
        dummy_context = torch.zeros(B, flow.context_dim) if flow.context_dim > 0 else None
        
        y, _ = flow(x_in, dummy_context)
        x_rec = flow.inverse(y, dummy_context)

        print("[SANITY CHECK] max |x - inverse(forward(x))| =",
              (x_in - x_rec).abs().max().item())
    
    # with torch.no_grad():
    #     B = 10
    #     # prendi un layer reale dal flow
    #     layer = flow.layers[0]
 
    #     D = flow.dim
    #     context_dim = context_encoder.output_dim
 
    #     x = torch.randn(B, D, device=device)
        
    #     context = torch.randn(10, context_dim, device=device)
 
    #     y, _ = layer(x, context)
    #     x_rec = layer.inverse(y, context)
 
    #     max_err = (x - x_rec).abs().max().item()
    #     print(f"[SANITY CHECK] max |x - inverse(forward(x))| = {max_err:.3e}")
 
    #     assert max_err < 1e-6, "Coupling layer is NOT invertible!"

class ConditionalFlow(nn.Module):
    def __init__(self, dim, n_layers, hidden_dim, masks, context_dim=0):
        """
        dim        : dimensione delle feature da trasformare
        n_layers   : numero di coupling layers
        hidden_dim : dimensione hidden network di ciascun coupling
        masks      : lista di mask tensor (uno per layer)
        context_dim: dimensione del context condizionale
        """
        super().__init__()
        assert len(masks) == n_layers, "Serve una mask per ogni layer"
        self.dim=dim
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        self.context_dim=context_dim
        self.masks=masks
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(
                dim=dim,
                context_dim=context_dim,
                mask=masks[i],
                hidden_dim=hidden_dim
            )
            for i in range(n_layers)
        ])
        self.dim = dim  # utile per sanity check

    def forward(self, x, context=None):
        """
        Applica il flow ai dati x con context condizionale
        x: (B, dim_obs + dim_latent)  # dim_latent opzionale
        context: (B, context_dim) o (B*N, context_dim)
        Ritorna: y, log_det totale
        """
        if x.shape[-1] < self.dim:
            raise ValueError("Input dimension smaller than flow dim")
        log_det_total = 0
        y = x
        for layer in self.layers:
            y, log_det = layer(y, context)
            log_det_total += log_det
        return y, log_det_total

    def inverse(self, z, context=None):
        """
        Inverte il flow
        z: (B, dim) o (B, N, dim)
        context: stesso context usato nel forward
        """
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, context)
        return x

    # Chiave per checkpoint
    def get_config(self):
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "masks": self.masks,
        }

def sanity_check_flow(flow, context_encoder, device="cpu"):
    flow.eval()
    context_encoder.eval()
 
    with torch.no_grad():
        x = torch.randn(10, flow.dim, device=device)
        raw_context = torch.randn(10, context_encoder.input_dim, device=device)
        context = context_encoder(raw_context)
 
        z, _ = flow(x, context)
        x_rec = flow.inverse(z, context)
 
        max_err = (x - x_rec).abs().max().item()
        print(f"[FLOW CHECK] max reconstruction error = {max_err:.3e}")
 
        assert max_err < 1e-6

class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers=2,
        activation=nn.ReLU,
        dropout=0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (N, input_dim) = (alpha, beta, x, y)
        """
        return self.net(x)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
        }

def gaussian_kernel(x, y, sigma):
    """
    x: (N, d)
    y: (M, d)
    """
    x = x.unsqueeze(1)  # (N, 1, d)
    y = y.unsqueeze(0)  # (1, M, d)
    dist2 = ((x - y) ** 2).sum(2)
    return torch.exp(-dist2 / (2 * sigma ** 2))


def mmd_loss(x, y, sigma=1.0, sigma_datadriven=True):
    if sigma_datadriven:
        pairwise = torch.cdist(x.detach(), y.detach())
        sigma = torch.median(pairwise)
    Kxx = gaussian_kernel(x, x, sigma).mean()
    Kyy = gaussian_kernel(y, y, sigma).mean()
    Kxy = gaussian_kernel(x, y, sigma).mean()
    return Kxx + Kyy - 2 * Kxy

def conditional_mmd(A_corr, A_data, min_events=20, sigma=1.0, sigma_datadriven=True):
    """
    A_corr: (N1, D)
    A_data: (N2, D)
    """
    if A_corr.shape[0] < min_events or A_data.shape[0] < min_events:
        return None

    return mmd_loss(
        A_corr,
        A_data,
        sigma=sigma,
        sigma_datadriven=sigma_datadriven,
    )

def identity_loss(flow, A_src, context, weight=1.0):
    """
    Penalizza deviazioni quando la trasformazione dovrebbe
    essere prossima all'identità.
    """
    A_id, _ = flow(A_src, context)
    return weight * ((A_id - A_src) ** 2).mean()

def subsample(A, n, device="cpu"):
    N = A.shape[0]
    if N >= n:
        idx = torch.randperm(N, device=device)[:n]
    else:
        idx = torch.randint(0, N, (n,), device=device)
    return A[idx]

def subsample_dynamic(A, n_max, device="cpu"):
    """
    A: torch.Tensor (B, N, D) oppure (N, D)
    n_max: numero massimo di eventi desiderato
    """
    if A.ndim == 2:  # (N, D)
        N = A.shape[0]
        if N >= n_max:
            idx = torch.randperm(N, device=device)[:n_max]
        else:
            idx = torch.arange(N, device=device)
        return A[idx]

    elif A.ndim == 3:  # (B, N, D)
        B, N, D = A.shape
        if N >= n_max:
            idx = torch.randperm(N, device=device)[:n_max]
        else:
            idx = torch.arange(N, device=device)
        # sottocampiona lungo la dimensione 1 (eventi)
        return A[:, idx, :]
    
    else:
        raise ValueError(f"Unsupported tensor shape: {A.shape}")
    
def train_step(
        step,
        flow,
        context_encoder,
        optimizer,
        batch,
        n_events=50,
        lambda_mom=0.1,
        test_identity=False,
        device="cpu"):
    """
    Step di training / identity test per un batch.
    flow: ConditionalFlow
    context_encoder: ContextEncoder
    optimizer: torch optimizer
    batch: dict con 'sim_key', 'data_key', 'A_sim', 'A_data'
    n_events: numero di eventi da sottocampionare
    lambda_mom: peso ID loss
    test_identity: se True usa A_src = A_data
    device: torch device
    """

    flow.train()
    context_encoder.train()
    optimizer.zero_grad()

    assert batch["A_sim"].shape[-1] == flow.dim
    
    # --- selezione dati ---
    ## EDM prima
    ##A_sim_full = batch["A_sim"] #.to(device)
    ##A_data_full = batch["A_data"] #.to(device)

    A_sim_full  = batch["A_sim"]
    A_data_full = batch["A_data"]

    # rimuovi dimensione batch fittizia
    A_sim_full  = batch["A_sim"].squeeze(0)    # (Ns, D)
    A_data_full = batch["A_data"].squeeze(0)   # (Nd, D)

    # NB. Il DataLoader prende solo 1 chiave per batch (batch_size=1) => 1 sola chiave per sim, e 1 per data. Quindi posso normalizzare / chiave
    sim_key  = batch["sim_key"].view(-1)     # (Cs,) # view(-1) serve a prendere solo l'array piatto del batch, perche' il batch viene dal DataLoader
    data_key = batch["data_key"].view(-1)    # (Cd,)

    # === REMOVE OBSERVED Z FROM THE CONTEXT, TO FORCE 1:1 MAPPING BETWEEN SIM AND DATA ===
    data_key_reduced = data_key[..., 1:]
    
    # standardizzazione
    mu_sim = batch["sim_mu"].view(-1)
    std_sim = batch["sim_std"].view(-1)
    mu_data = batch["data_mu"].view(-1)
    std_data = batch["data_std"].view(-1)
    if step==0:
        #print("Tensor full = ",A_sim_full)
        print(f"Before standardization A_sim_full: mu={mu_sim},std={std_sim}")
        print(f"Before standardization A_data_full: mu={mu_data},std={std_data}")
    A_sim_full_scaled = standardize(A_sim_full,mu_sim,std_sim)
    A_data_full_scaled = standardize(A_data_full,mu_data,std_data)
    if step==0:
        #print("Tensor scaled = ",A_sim_full)
        mu_aft=A_sim_full_scaled.mean(dim=0)
        std_aft=A_sim_full_scaled.std(dim=0)
        print(f"After standardization, A_sim_full (scaled): mu={mu_aft},std={std_aft}")
        print(f"After standardization  A_data_full (scaled): mu={A_data_full_scaled.mean(0)},std={A_data_full_scaled.std(0)}")
        
    # subsampling coerente dei dataset in batch di n_events ciascuno
    N = min(len(A_sim_full_scaled), len(A_data_full_scaled), n_events)
    A_sim_sub  = subsample_dynamic(A_sim_full_scaled, N, device=A_sim_full.device)
    A_data_sub = subsample_dynamic(A_data_full_scaled, N, device=A_data_full.device)
    
    if A_sim_sub.shape[1] == 0:
        # fallback: copia tutti gli eventi disponibili
        A_sim_sub = A_sim_full_scaled.clone()
    if A_data_sub.shape[1] == 0:
        A_data_sub = A_data_full_scaled.clone()

    sigma_latent = 5.0 # scala circa pari a std dei dati nello spazio standardizzato
    z_latent = sigma_latent * torch.randn_like(A_sim_sub)
    # Combina input con latent
    A_corr_input = torch.cat([A_sim_sub, z_latent], dim=-1)

    # --- Costruzione del contesto ---
    # --- fundamental (previous bug): context event by event, so for each batch, the context
    #     (alpha,beta,x,y) is always the same
    context_vals = torch.cat([sim_key, data_key_reduced], dim=0)  # (C=Cs+Cd,)

    context = context_vals.unsqueeze(0).expand(N, -1)  # (N,C)
    assert context_vals.dim() == 1, context_vals.shape
    assert context.shape == (N, context_vals.numel())
    cond = context_encoder(context)

    # --- forward pass ---
    A_corr_scaled, _ = flow(A_corr_input, cond)    
    # print("A_sim_full mean/std:",
    #       A_sim_full.mean().item(),
    #       A_sim_full.std().item())
    # print("A_sim_full_scaled mean/std:",
    #       A_sim_full_scaled.mean().item(),
    #       A_sim_full_scaled.std().item())
    # print("post-flow mean/std:",
    #       A_corr_scaled.mean().item(),
    #       A_corr_scaled.std().item())

    # if math.isnan(A_sim_full.std().item()):
    #     print ("Problematic tensor  = ",A_sim_full)
    
    # --- calcolo loss ---
    if test_identity:
        # Identity loss
        id_loss = torch.mean((A_corr_scaled - A_sim_sub)**2)
        loss_mmd = torch.tensor(0.0, device=device)
    else:
        if A_sim_sub.shape[1] == 0 or A_data_sub.shape[1] == 0:
            # salta questo batch
            return {
                "loss": None,
                "loss_mmd": None,
                "moment_loss": None,
                "skip": True
            }
        id_loss = torch.tensor(0.0, device=A_corr_scaled.device)

        n_data_feat = A_data_sub.shape[1]
        A_corr_for_mmd = A_corr_scaled[:, :n_data_feat]  # ignora latent in coda
        loss_mmd = conditional_mmd(A_corr_for_mmd, A_data_sub, min_events=10)

        if loss_mmd is None:
            return {
                "loss": None,
                "loss_mmd": None,
                "moment_loss": None,
                "skip": True
            }

    mu_corr  = A_corr_scaled.mean(0)
    std_corr = A_corr_scaled.std(0, unbiased=False)
    mu_data_sub = A_data_sub.mean(0)
    std_data_sub  = A_data_sub.std(0, unbiased=False)

    mean_loss = (mu_corr   - mu_data_sub ).pow(2).mean()
    logstd_loss = (torch.log(std_corr + 1e-8) - torch.log(std_data_sub + 1e-8)).pow(2).mean() 
    moment_loss = mean_loss + logstd_loss
    #rms_loss = (torch.log(std_corr / std_data_sub)).pow(2)
    total_loss = loss_mmd + lambda_mom * moment_loss
    #total_loss = loss_mmd
    
    # --- backward ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        "loss": total_loss.item(),
        "loss_mmd": loss_mmd.item(),
        "moment_loss": moment_loss.item(),
        "skip": False
    }

def compute_val_mmd(flow, context_encoder, val_case, n_events):

    # Subsampling uniforme sulla distribuzione di validazione
    A_sim  = subsample_dynamic(val_case["A_sim"],  n_events)
    A_data = subsample_dynamic(val_case["A_data"], n_events)

    context = val_case["context"].repeat(len(A_sim), 1)

    # --- inference senza grad ---
    with torch.no_grad():
        cond = context_encoder(context)
        A_corr, _ = flow(A_sim, cond)

    # --- loss --- 
    loss = mmd_loss(A_corr, A_data)
    return loss

def standardize(x, mu, std):
    return (x - mu) / std

class SimulationCorrection():

    def __init__(self, configuration, dataset,standardize,
                 encoder_input_dim, encoder_hidden_dim, encoder_output_dim, encoder_n_layers, encoder_dropout,
                 flow_n_layers, flow_hidden_dim, flow_context_dim,
                 initial_lr, batch_size, lambda_mom):

        # Name of the variables used as conditions and during training
        self.dataset = dataset
        self.standardize = standardize
        
        # Checking if cuda is avaliable
        print('Checking cuda avaliability: ', torch.cuda.is_available())
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.encoder_input_dim = encoder_input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.encoder_n_layers = encoder_n_layers
        self.encoder_dropout = encoder_dropout

        self.flow_n_layers = flow_n_layers
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_context_dim = flow_context_dim
        
        # general training hyperparameters
        self.initial_lr       = initial_lr
        self.batch_size       = batch_size
        self.lambda_mom        = lambda_mom

        # Now, lets open a directory to store the results and models of a given configuration
        self.configuration =  configuration
        #lets create a folder with the results
        try:
            print('\nThis run dump folder: ', os.getcwd() + '/results/' + self.configuration + '/')
            #already creating the folders to store the flow states and the plots
            os.makedirs(os.getcwd() + '/results/' +self.configuration + '/',  exist_ok=True)
            os.makedirs(os.getcwd() + '/results/' +self.configuration + '/saved_states/',  exist_ok=True)
        except:
            print('\nIt was not possible to open the dump folder')
            exit()

        # folder to which the code will store the results
        self.dump_folder = os.getcwd() + '/results/' +self.configuration + '/'

    # performs the training of the normalizing flows
    def setup_flow(self):

        self.D = next(iter(self.dataset))["A_sim"].shape[-1] # numero di variabili

        self.masks = generate_alternating_masks(self.D, self.flow_n_layers)

        self.flow = ConditionalFlow(dim=self.D,
                                    n_layers=self.flow_n_layers,
                                    hidden_dim=self.flow_hidden_dim,
                                    masks=self.masks,
                                    context_dim=self.flow_context_dim).to(self.device)

        assert self.flow.dim == self.D
        
        self.context_encoder = ContextEncoder(input_dim=self.encoder_input_dim, hidden_dim=self.encoder_hidden_dim, output_dim=self.encoder_output_dim, n_layers=self.encoder_n_layers, dropout=self.encoder_dropout).to(self.device)
        sanity_check_coupling(self.flow, self.context_encoder, device=self.device)
        sanity_check_flow(self.flow, self.context_encoder, device=self.device)

        self.optimizer = torch.optim.Adam(list(self.flow.parameters()) + list(self.context_encoder.parameters()), lr=self.initial_lr)

    def set_validation_case(self, datasets_and_context):
        self.val_case = datasets_and_context

        
    def train_the_flow(self, test_identity=False):

        loader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        # Training
        print("Now training the flow...")
        val_every  = 200     # ogni 200 step
        patience   = 800     # ~4 validazioni senza migliorare
        min_delta  = 0.005   # soglia reale (rumore MMD)
        best_val_mmd = float("inf")
        best_step = 0
        # --- Ottimizzazione del training: ---
        
        for step, batch in enumerate(loader):

            losses = train_step(step, self.flow, self.context_encoder, self.optimizer, batch, n_events=self.batch_size, lambda_mom=self.lambda_mom,test_identity=test_identity,device=self.device)
            if losses.get("skip", False):
                continue
            if step % 100 == 0:
                print(
                    f"step {step:5d} | "
                    f"MMD {losses['loss_mmd']:.4f} | "
                    f"MOMENT {losses['moment_loss']:.4f}"
                )
            # if step == 100:
            #     with torch.no_grad():
            #         ### sto qua
            #         #A_corr1 = flow(A_sim_sub, cond1)
            #         #A_corr2 = flow(A_sim_sub, cond2)

            #         print("Δ corr:", torch.mean(torch.abs(A_corr1 - A_corr2)))
                
            # ---- VALIDAZIONE PERIODICA ----
            # ---- qui l'implementazione dell'Early Stopping --- 
            if step % val_every == 0 and step > 0:

                val_mmd = compute_val_mmd(self.flow, self.context_encoder, self.val_case, self.batch_size)
         
                print(f"  → Validation MMD = {val_mmd:.4f}")
         
                if val_mmd < best_val_mmd - min_delta:
                    best_val_mmd = val_mmd
                    best_step = step

                    # Save the output
                    torch.save({
                        "flow_state": self.flow.state_dict(),
                        "context_state": self.context_encoder.state_dict(),
                        "flow_config": self.flow.get_config(),
                        "context_config": self.context_encoder.get_config(),
                        "best_step": step,
                        "best_val_mmd": best_val_mmd,
                        "lambda_mom": self.lambda_mom
                    }, os.getcwd() + "/results/" + self.configuration + "/saved_states/best_model.pt")
         
                    print(f"  ✓ new best model at step {step}")
         
                elif step - best_step > patience:
                    print(
                        f"Early stopping at step {step} "
                        f"(best step {best_step}, val MMD {best_val_mmd:.4f})"
                    )
                    break
                   
def load_model(
    checkpoint_path,
    device="cpu",
    strict=True,
):
    """
    Carica ConditionalFlow + ContextEncoder da checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path a best_model.pt
    device : str or torch.device
        'cpu' o 'cuda'
    strict : bool
        Passato a load_state_dict

    Returns
    -------
    flow : ConditionalFlow
    context_encoder : ContextEncoder
    metadata : dict
        Info utili (step, val_mmd, ecc.)
    """

    ckpt = torch.load(checkpoint_path, map_location=device)

    # ---- sanity checks ----
    required_keys = [
        "flow_state",
        "context_state",
        "flow_config",
        "context_config",
    ]
    for k in required_keys:
        if k not in ckpt:
            raise KeyError(f"Checkpoint incompleto: manca '{k}'")

    # ---- ricostruzione modelli ----
    flow = ConditionalFlow(**ckpt["flow_config"])
    flow.load_state_dict(ckpt["flow_state"], strict=strict)
    flow.to(device)
    flow.eval()

    context_encoder = ContextEncoder(**ckpt["context_config"])
    context_encoder.load_state_dict(ckpt["context_state"], strict=strict)
    context_encoder.to(device)
    context_encoder.eval()

    # ---- metadata utile ----
    metadata = {
        k: ckpt[k]
        for k in ckpt.keys()
        if k not in [
            "flow_state",
            "context_state",
            "flow_config",
            "context_config",
        ]
    }

    return flow, context_encoder, metadata

# this is to check that the flow is correctly closing (DEBUG, do just once)
def atomic_flow_test(flow, dim_x, dim_c, device="cpu"):
    flow.eval()

    with torch.no_grad():
        x = torch.randn(1000, dim_x, device=device)

        # contesto FISSO e controllato
        c = torch.zeros(1000, dim_c, device=device)

        y, log_det = flow(x, c)
        x_rec = flow.inverse(y, c)

        max_err = torch.max(torch.abs(x - x_rec))
        print(f"[ATOMIC TEST] max |x - inverse(forward(x))| = {max_err:.3e}")

        assert max_err < 1e-5, "Atomic test FAILED"

def generate_alternating_masks(dim, n_layers):
    """
    Crea una lista di mask per ConditionalAffineCoupling
    dim      : numero di feature del tuo flow
    n_layers : numero di coupling layers
    Ritorna  : lista di torch.Tensor di shape (dim,)
    """
    masks = []
    for i in range(n_layers):
        mask = torch.zeros(dim)
        # alterna 1 e 0 con shift layer per layer
        mask[i % 2::2] = 1
        masks.append(mask)
    return masks





### da mettere in un py file separato

# A_sim, A_data: torch.Tensor (N_events, N_features)
# A_corr: output del flow (N_events, N_features)
# Assicurati che siano tutti float32 e sullo stesso device

def print_numeric_validation(A_sim,A_data,A_corr):
    # Trasforma eventuali batch in (N, D)
    if A_sim.ndim == 3:
        A_sim = A_sim.squeeze(0)
    if A_data.ndim == 3:
        A_data = A_data.squeeze(0)
    if A_corr.ndim == 3:
        A_corr = A_corr.squeeze(0)


    mean_sim = A_sim.mean(0)
    std_sim  = A_sim.std(0)

    mean_corr = A_corr.mean(0)
    std_corr  = A_corr.std(0)

    mean_data = A_data.mean(0)
    std_data  = A_data.std(0)

    mmd = mmd_loss(A_corr, A_data)
    print("MMD in scaled space:", mmd.item())
    print("means:", A_corr.mean().item(), A_data.mean().item())
    print("stds :", A_corr.std().item(),  A_data.std().item())
    
    ks = scipy.stats.ks_2samp(
        A_corr.cpu().numpy().ravel(),
        A_data.cpu().numpy().ravel()
    ).statistic
    
    print("Feature-wise mean (scaled):")
    print("sim  :", mean_sim)
    print("corr :", mean_corr)
    print("data :", mean_data)

    print("\nFeature-wise std (scaled):")
    print("sim  :", std_sim)
    print("corr :", std_corr)
    print("data :", std_data)

    print ("Distance metrics:")
    print("mmd  :", mmd)
    print("ks   :", ks)
    
def standardize_dataset(A):
    mean  = A.mean(0)
    std   = A.std(0) + 1e-8
    print(f"validation standardization: mean = {mean}, std = {std}")
    A = (A - mean) / std
    return (A,mean,std)
