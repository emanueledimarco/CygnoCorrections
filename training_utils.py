# python libraries import
import os 
import numpy as np
import glob
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

        mask = self.mask.to(x.device)
        x_masked = x_flat[:, mask.bool()]

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
        y_flat[:, (~mask.bool())] = x_flat[:, (~mask.bool())] * torch.exp(s) + t

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

        mask = self.mask.to(y.device)
        y_masked = y_flat[:, mask.bool()]

        if context is not None:
            context_flat = context.view(y_masked.shape[0], -1)
            net_input = torch.cat([y_masked, context_flat], dim=1)
        else:
            net_input = y_masked

        s, t = self.st_net(net_input).chunk(2, dim=1)
        s = torch.tanh(s)

        x_flat = y_flat.clone()
        x_flat[:, (~mask.bool())] = (y_flat[:, (~mask.bool())] - t) * torch.exp(-s)

        x = x_flat.view(*orig_shape)
        log_det = -s.sum(dim=1)

        return x    

def sanity_check_coupling(flow, context_encoder, device="cpu"):
    flow.eval()
    context_encoder.eval()
 
    with torch.no_grad():
        # prendi un layer reale dal flow
        layer = flow.layers[0]
 
        D = flow.dim
        context_dim = context_encoder.output_dim
 
        x = torch.randn(10, D, device=device)
        context = torch.randn(10, context_dim, device=device)
 
        y, _ = layer(x, context)
        x_rec = layer.inverse(y, context)
 
        max_err = (x - x_rec).abs().max().item()
        print(f"[SANITY CHECK] max |x - inverse(forward(x))| = {max_err:.3e}")
 
        assert max_err < 1e-6, "Coupling layer is NOT invertible!"

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
        x: (B, dim) o (B, N, dim)
        context: (B, context_dim) o (B*N, context_dim)
        Ritorna: y, log_det totale
        """
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
    flow,
    context_encoder,
    optimizer,
    batch,
    n_events=50,
    lambda_id=0.1,
    sigma_mmd=1.0,
    test_identity=False,
    device="cpu"
):
    """
    Step di training / identity test per un batch.
    flow: ConditionalFlow
    context_encoder: ContextEncoder
    optimizer: torch optimizer
    batch: dict con 'sim_key', 'data_key', 'A_sim', 'A_data'
    n_events: numero di eventi da sottocampionare
    lambda_id: peso ID loss
    sigma_mmd: parametro del kernel MMD
    test_identity: se True usa A_src = A_data
    device: torch device
    """

    flow.train()
    context_encoder.train()
    optimizer.zero_grad()

    # --- selezione dati ---
    A_sim_full = batch["A_sim"].to(device)
    A_data_full = batch["A_data"].to(device)

    # subsampling dei dataset in batch di n_events ciascuno
    N = min(n_events, A_sim_full.shape[0], A_data_full.shape[0])
    A_sim_sub  = subsample_dynamic(A_sim_full, N, device=A_sim_full.device)
    A_data_sub = subsample_dynamic(A_data_full, N, device=A_data_full.device)

    if A_sim_sub.shape[1] == 0:
        # fallback: copia tutti gli eventi disponibili
        A_sim_sub = A_sim_full.clone()
    if A_data_sub.shape[1] == 0:
        A_data_sub = A_data_full.clone()

    # --- Costruzione del contesto ---
    context = build_context_batch(batch, test_identity=test_identity, device=device)
    cond = context_encoder(context)

    # --- forward pass ---
    A_corr, _ = flow(A_sim_sub, cond)

    # --- calcolo loss ---
    if test_identity:
        # Identity loss
        id_loss = torch.mean((A_corr - A_sim_sub)**2)
        loss_mmd = torch.tensor(0.0, device=device)
    else:
        if A_sim_sub.shape[1] == 0 or A_data_sub.shape[1] == 0:
            # salta questo batch
            return {
                "loss": None,
                "loss_mmd": None,
                "loss_id": None,
                "skip": True
            }
        # MMD loss solo se non identity test
        if A_data_sub.shape[1] == 0:
            id_loss = torch.tensor(0.0, device=A_corr.device)
            loss_mmd = torch.tensor(0.0, device=device)
        else:
            id_loss = torch.tensor(0.0, device=A_corr.device)
            loss_mmd = mmd_loss(A_corr, A_data_sub, sigma=sigma_mmd, sigma_datadriven=True)

    total_loss = lambda_id * id_loss + loss_mmd
    
    # --- backward ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        "loss": total_loss.item(),
        "loss_mmd": loss_mmd.item(),
        "loss_id": id_loss.item(),
        "skip": False
    }

def compute_val_mmd(flow, context_encoder, val_case, n_events, sigma_mmd=1.0):

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

class SimulationCorrection():

    def __init__(self, configuration, dataset,
                 encoder_input_dim, encoder_hidden_dim, encoder_output_dim, encoder_n_layers, encoder_dropout,
                 flow_n_layers, flow_hidden_dim, flow_context_dim,
                 initial_lr, batch_size, sigma_mmd, lambda_id):

        # Name of the variables used as conditions and during training
        self.dataset = dataset
        
        # if False, the inputs are not standardized!
        self.perform_std_transform     = False
        print( 'Standardization: ', self.perform_std_transform )

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
        self.sigma_mmd        = sigma_mmd
        self.lambda_id        = lambda_id

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

        self.D = next(iter(self.dataset))["A_sim"].shape[1] # numero di variabili

        self.masks = generate_alternating_masks(self.D, self.flow_n_layers)

        self.flow = ConditionalFlow(dim=self.D,
                                    n_layers=self.flow_n_layers,
                                    hidden_dim=self.flow_hidden_dim,
                                    masks=self.masks,
                                    context_dim=self.flow_context_dim).to(self.device)
        
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
        #
        # sigma_mmd ≈ RMS delle distanze in A_sim
        # Troppo piccolo → overfitting locale

        # Troppo grande → perde struttura
        sigma_mmd = 1
        # lambda_id 
        # 0.01 → molto permissivo
        # 0.1  → buon default
	# 1.0  → molto conservativo
        lambda_id=1.0
        
        for step, batch in enumerate(loader):

            losses = train_step(self.flow, self.context_encoder, self.optimizer, batch, n_events=self.batch_size, lambda_id=self.lambda_id, sigma_mmd=self.sigma_mmd,test_identity=test_identity,device=self.device)
            if losses.get("skip", False):
                continue
            if step % 100 == 0:
                print(
                    f"step {step:5d} | "
                    f"MMD {losses['loss_mmd']:.4f} | "
                    f"ID {losses['loss_id']:.4f}"
                )
         
            # ---- VALIDAZIONE PERIODICA ----
            # ---- qui l'implementazione dell'Early Stopping --- 
            if step % val_every == 0 and step > 0:

                val_mmd = compute_val_mmd(self.flow, self.context_encoder, self.val_case, self.batch_size, sigma_mmd=self.sigma_mmd)
         
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
                        "sigma_mmd": self.sigma_mmd,
                        "lambda_id": self.lambda_id
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

def build_context_batch(batch, test_identity=False, device='cpu'):
    """
    Costruisce il tensore context batch robustamente.
    
    batch: dict con chiavi 'sim_key' e 'data_key'
    test_identity: se True, prende sim_key come source e target
    device: 'cpu' o 'cuda'
    
    Ritorna: context di shape (n_events, n_context_features)
    """
    # Scegli le chiavi
    key_src = batch["data_key"] if not test_identity else batch["sim_key"]
    key_tgt = batch["data_key"] if not test_identity else batch["sim_key"]

    # Se è un tensore scalare o ha più dimensioni, riducilo a 1D
    key_src = key_src.flatten()
    key_tgt = key_tgt.flatten()

    # Concateniamo le feature di contesto
    context = torch.cat([key_src, key_tgt], dim=0)  # (n_features,)

    # Trasformiamo in batch ripetendo per ogni evento
    n_events = batch["A_sim"].shape[0]
    context = context.unsqueeze(0).expand(n_events, -1)  # (n_events, n_features)

    return context.to(device)

import torch

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

    # Media e deviazione standard
    mean_sim  = A_sim.mean(0)
    std_sim   = A_sim.std(0)

    mean_corr = A_corr.mean(0)
    std_corr  = A_corr.std(0)

    mean_data = A_data.mean(0)
    std_data  = A_data.std(0)

    print("Feature-wise mean:")
    print("sim  :", mean_sim)
    print("corr :", mean_corr)
    print("data :", mean_data)

    print("\nFeature-wise std:")
    print("sim  :", std_sim)
    print("corr :", std_corr)
    print("data :", std_data)
