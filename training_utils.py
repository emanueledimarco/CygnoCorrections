# python libraries import
import os 
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim, context_dim, mask):
        super().__init__()

        assert dim == 2, "Qui assumiamo A bidimensionale"

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.mask = mask  # 0 o 1

        self.net = nn.Sequential(
            nn.Linear(1 + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # scale, shift
        )

    def forward(self, x, context):
        # x: (N, 2)
        # context: (N, context_dim)

        if self.mask == 0:
            x_id, x_tr = x[:, :1], x[:, 1:]
        else:
            x_tr, x_id = x[:, :1], x[:, 1:]

        h = torch.cat([x_id, context], dim=1)
        s, t = self.net(h).chunk(2, dim=1)

        s = torch.tanh(s)  # stabilità

        y_tr = x_tr * torch.exp(s) + t

        if self.mask == 0:
            y = torch.cat([x_id, y_tr], dim=1)
        else:
            y = torch.cat([y_tr, x_id], dim=1)

        return y

class ConditionalFlow(nn.Module):
    def __init__(self, dim, n_layers, hidden_dim, context_dim):
        super().__init__()

        self.dim = dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(
                dim=dim,
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                mask=i % 2
            )
            for i in range(n_layers)
        ])

    def forward(self, x, context):
        for layer in self.layers:
            x = layer(x, context)
        return x

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

def train_step(
    flow,
    context_encoder,
    optimizer,
    batch,
    lambda_id=0.1,
    sigma_mmd=1.0,
):
    """
    batch deve contenere:
      - A_sim:  (B, 2)
      - A_data: (B, 2)
      - context: (B, 4) = (alpha, beta, x, y)
    """

    flow.train()
    context_encoder.train()
    optimizer.zero_grad()

    A_sim  = batch["A_sim"]
    A_data = batch["A_data"]
    context_raw = batch["context"]

    # --- encode context ---
    context = context_encoder(context_raw)

    # --- forward ---
    A_corr = flow(A_sim, context)

    # --- losses ---
    loss_mmd = mmd_loss(A_corr, A_data, sigma=sigma_mmd)

    # identity: contesto nullo → trasformazione ~ identità
    zero_context = torch.zeros_like(context)
    loss_id = identity_loss(flow, A_sim, zero_context)

    loss = loss_mmd + lambda_id * loss_id

    # --- backward ---
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_mmd": loss_mmd.item(),
        "loss_id": loss_id.item(),
    }

@torch.no_grad()
def compute_val_mmd(flow, context_encoder, val_case, sigma_mmd=1.0):
    flow.eval()
    context_encoder.eval()

    A_sim = torch.tensor(val_case["A_sim"], dtype=torch.float32)
    A_data = torch.tensor(val_case["A_data"], dtype=torch.float32)

    context = val_case["context"].repeat(len(A_sim), 1)
    context = context_encoder(context)

    A_corr = flow(A_sim, context)

    # sottocampionamento per stabilità
    n = min(1000, len(A_corr), len(A_data))
    return mmd_loss(A_corr[:n], A_data[:n], sigma=sigma_mmd).item()


class SimulationCorrection():

    def __init__(self, configuration, dataset,
                 encoder_input_dim, encoder_hidden_dim, encoder_output_dim, encoder_n_layers, encoder_dropout,
                 flow_n_dim, flow_n_layers, flow_hidden_dim, flow_context_dim,
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

        # defining the flow hyperparametrs as members of the class
        self.encoder_input_dim = encoder_input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.encoder_n_layers = encoder_n_layers
        self.encoder_dropout = encoder_dropout

        self.flow_n_dim = flow_n_dim
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

        self.context_encoder = ContextEncoder(input_dim=self.encoder_input_dim, hidden_dim=self.encoder_hidden_dim, output_dim=self.encoder_output_dim, n_layers=self.encoder_n_layers, dropout=self.encoder_dropout)
        
        self.flow = ConditionalFlow(dim=self.flow_n_dim, n_layers=self.flow_n_layers, hidden_dim=self.flow_hidden_dim, context_dim=self.flow_context_dim)

        self.optimizer = torch.optim.Adam(list(self.flow.parameters()) + list(self.context_encoder.parameters()), lr=self.initial_lr)

    def set_validation_case(self, datasets_and_context):
        self.val_case = datasets_and_context
        
    def train_the_flow(self):

        loader = DataLoader(self.dataset, batch_size=None, shuffle=True)

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

            losses = train_step(self.flow, self.context_encoder, self.optimizer, batch, lambda_id=self.lambda_id, sigma_mmd=self.sigma_mmd)
         
            if step % 100 == 0:
                print(
                    f"step {step:5d} | "
                    f"MMD {losses['loss_mmd']:.4f} | "
                    f"ID {losses['loss_id']:.4f}"
                )
         
            # ---- VALIDAZIONE PERIODICA ----
            # ---- qui l'implementazione dell'Early Stopping --- 
            if step % val_every == 0 and step > 0:
                val_mmd = compute_val_mmd(self.flow, self.context_encoder, self.val_case, sigma_mmd=self.sigma_mmd)
         
                print(f"  → val MMD = {val_mmd:.4f}")
         
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
