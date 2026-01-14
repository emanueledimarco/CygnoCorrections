# python libraries import
import os 
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, context_dim, hidden_dim=128, mask=None):
        super().__init__()
        self.dim = dim

        if mask is None:
            mask = torch.zeros(dim)
            mask[::2] = 1  # alterna 1 e 0
        self.register_buffer("mask", mask)

        in_dim = int(mask.sum()) + context_dim
        out_dim = dim - int(mask.sum())

        self.st_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * out_dim)
        )

    def forward(self, x, context=None):
        # x: [batch, dim], context: [batch, dim_context]
        mask = self.mask  # [dim]
        
        # Maschera e parte passante
        x_masked = x * mask           # feature mascherate → input di s,t
        x_pass = x * (1 - mask)      # feature che verranno trasformate
     
        # Calcolo s e t solo sulle feature non mascherate
        st_input = torch.cat([x_masked, context], dim=1) if context is not None else x_masked
        st = self.st_net(st_input)  # [batch, 2 * dim_unmasked]
        
        # Separiamo s e t
        dim_unmasked = int((1 - mask).sum().item())
        s = st[:, :dim_unmasked]
        t = st[:, dim_unmasked:]
     
        # Applichiamo la trasformazione affine solo alle feature non mascherate
        x_transformed = x_pass.clone()
        if dim_unmasked > 0:
            x_transformed[:, mask == 0] = x_pass[:, mask == 0] * torch.exp(s) + t
     
        # Combiniamo con la parte mascherata
        y = x_masked + x_transformed
     
        # Log-det Jacobiano
        log_det = s.sum(dim=1) if dim_unmasked > 0 else torch.zeros(x.size(0), device=x.device)
     
        return y, log_det


    def inverse(self, y, context=None):
        mask = self.mask  # [dim]
     
        # Separiamo le feature mascherate e non mascherate
        y_masked = y * mask
        y_pass = y * (1 - mask)
     
        # Calcolo s e t come nel forward
        st_input = torch.cat([y_masked, context], dim=1) if context is not None else y_masked
        st = self.st_net(st_input)
     
        dim_unmasked = int((1 - mask).sum().item())
        s = st[:, :dim_unmasked]
        t = st[:, dim_unmasked:]
     
        # Inverse affine sulle feature non mascherate
        x_pass = y_pass.clone()
        if dim_unmasked > 0:
            x_pass[:, mask == 0] = (y_pass[:, mask == 0] - t) * torch.exp(-s)
     
        # Combiniamo con la parte mascherata
        x = y_masked + x_pass
     
        # Log-det Jacobiano (negativo di forward)
        log_det = -s.sum(dim=1) if dim_unmasked > 0 else torch.zeros(y.size(0), device=y.device)
     
        return x, log_det


class ConditionalFlow(nn.Module):
    def __init__(self, dim, context_dim, n_layers=6, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        layers = []
        for i in range(n_layers):
            mask = self._alternating_mask(dim, i)
            layers.append(
                ConditionalAffineCoupling(
                    dim=dim,
                    context_dim=context_dim,
                    hidden_dim=hidden_dim,
                    mask=mask
                )
            )
        self.layers = nn.ModuleList(layers)

    def _alternating_mask(self, dim, i):
        mask = torch.zeros(dim)
        mask[i % 2::2] = 1
        return mask

    def forward(self, x, context):
        log_det = 0
        for layer in self.layers:
            x, ld = layer(x, context)
            log_det += ld
        return x, log_det

    def inverse(self, z, context):
        log_det = 0
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, context)
            log_det += ld
        return z, log_det
    
    # Chiave per checkpoint
    def get_config(self):
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
        }
    

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


def mmd_loss(x, y, sigma=1.0):
    Kxx = gaussian_kernel(x, x, sigma).mean()
    Kyy = gaussian_kernel(y, y, sigma).mean()
    Kxy = gaussian_kernel(x, y, sigma).mean()
    return Kxx + Kyy - 2 * Kxy

def identity_loss(flow, A_src, context, weight=1.0):
    """
    Penalizza deviazioni quando la trasformazione dovrebbe
    essere prossima all'identità.
    """
    A_id = flow(A_src, context)
    return weight * ((A_id - A_src) ** 2).mean()

def subsample(A, n):
    N = A.shape[0]
    if N >= n:
        idx = torch.randperm(N, device=A.device)[:n]
    else:
        idx = torch.randint(0, N, (n,), device=A.device)
    return A[idx]

def subsample_dynamic(A, n_max):
    """
    A: torch.Tensor (N, D)
    n_max: numero massimo di eventi desiderato
    """
    N = A.shape[0]
    
    # Se abbiamo più eventi di n_max → campiona senza replacement
    if N >= n_max:
        idx = torch.randperm(N, device=A.device)[:n_max]
    # Altrimenti usa tutti gli eventi
    else:
        idx = torch.arange(N, device=A.device)
    
    return A[idx]

def train_step(
    flow,
    context_encoder,
    optimizer,
    batch,
    n_events=50,
    lambda_id=0.1,
    sigma_mmd=1.0,
):
    """
    batch: dict contenente
        'A_sim': Tensor(N_s, D)
        'A_data': Tensor(N_t, D)
        'sim_key': tuple (alpha,beta)
        'data_key': tuple (x,y)
    """

    flow.train()
    context_encoder.train()
    optimizer.zero_grad()
    
    # N.B. Lo squeeze(0) serve per rimuovere la dimensione aggiuntiva al tensore aggiunta dal data-loader (come batch 1)
    A_sim_full  = batch["A_sim"].squeeze(0)
    A_data_full = batch["A_data"].squeeze(0)

    # subsampling dei dataset in batch di n_events ciascuno
    A_sim  = subsample_dynamic(A_sim_full, n_events)
    A_data = subsample_dynamic(A_data_full, n_events)

    # --- Costruzione del contesto ---
    sim_key = batch["sim_key"].squeeze(0)
    data_key = batch["data_key"].squeeze(0)
    print("sim key = ",sim_key)

    context_vals = list(sim_key) + list(data_key)   # lista di float
    print("cont vals =",context_vals)
    context_vals = torch.tensor(context_vals, dtype=torch.float32, device=A_sim.device)

    context = context_vals.unsqueeze(0).repeat(len(A_sim), 1)
    cond = context_encoder(context)

    print("src_key.shape:", sim_key.shape)
    print("tgt_key.shape:", sim_key.shape)
    print("context.shape:", context.shape)
    #print("x.shape:", x.shape)
    
    # --- forward pass ---
    A_corr = flow(A_sim, cond)

    # --- losses ---
    loss_mmd = mmd_loss(A_corr, A_data, sigma=sigma_mmd)

    # identity: contesto nullo → trasformazione ~ identità
    zero_context = torch.zeros_like(cond)
    loss_id = identity_loss(flow, A_sim, zero_context)

    loss = loss_mmd + lambda_id * loss_id

    # --- backward propagation ---
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_mmd": loss_mmd.item(),
        "loss_id": loss_id.item(),
    }

def compute_val_mmd(flow, context_encoder, val_case, n_events, sigma_mmd=1.0):

    # Subsampling uniforme sulla distribuzione di validazione
    A_sim  = subsample_dynamic(val_case["A_sim"],  n_events)
    A_data = subsample_dynamic(val_case["A_data"], n_events)
                
    # Costruzione contesto
    context_vals = list(val_case["sim_key"]) + list(val_case["data_key"])
    context = torch.tensor([context_vals]*len(A_sim), dtype=torch.float32, device=A_sim.device)

    # Forward
    with torch.no_grad():
        A_corr = flow(A_sim, cond)
    val_mmd = mmd_loss(A_corr, A_data, sigma=sigma_mmd).item()

    return val_mmd 


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

        self.flow = ConditionalFlow(dim=self.D, n_layers=self.flow_n_layers, hidden_dim=self.flow_hidden_dim, context_dim=self.flow_context_dim).to(self.device)

        self.context_encoder = ContextEncoder(input_dim=self.encoder_input_dim, hidden_dim=self.encoder_hidden_dim, output_dim=self.encoder_output_dim, n_layers=self.encoder_n_layers, dropout=self.encoder_dropout).to(self.device)

        self.optimizer = torch.optim.Adam(list(self.flow.parameters()) + list(self.context_encoder.parameters()), lr=self.initial_lr)

    def set_validation_case(self, datasets_and_context):
        self.val_case = datasets_and_context
        
    def train_the_flow(self):

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

            losses = train_step(self.flow, self.context_encoder, self.optimizer, batch, n_events=self.batch_size, lambda_id=self.lambda_id, sigma_mmd=self.sigma_mmd)
         
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
                        "lambda_id": self.lambda_id,
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
