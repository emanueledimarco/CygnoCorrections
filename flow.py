import numpy as np
import torch
import torch.nn as nn
import ROOT
import argparse

from torch.utils.data import DataLoader
from HistFlowDataset import *
from FlowValidation import TH2FlowValidation

def pair_grid(x,y,Ny):
    return x * Ny + y

def plot_dataset(DS,pname="A_out"):

    h_out = ROOT.TH2D("out", "out",
                      100, -3, 7,
                      100, -5, 5)
    
    A = DS.cpu().numpy()
    for x, y in A:
        h_out.Fill(x, y)

    c = ROOT.TCanvas("c","",600,600)
    h_out.Draw("colz")
    c.SaveAs(f"{pname}.pdf")


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


def save_arrays_root(fname, A_sim, A_corr, A_data):
    f = ROOT.TFile(fname, "RECREATE")
    t = ROOT.TTree("events", "Flow validation")

    import array
    xs = array.array('f', [0])
    ys = array.array('f', [0])
    xc = array.array('f', [0])
    yc = array.array('f', [0])
    xd = array.array('f', [0])
    yd = array.array('f', [0])

    t.Branch("sim_x", xs, "sim_x/F")
    t.Branch("sim_y", ys, "sim_y/F")
    t.Branch("corr_x", xc, "corr_x/F")
    t.Branch("corr_y", yc, "corr_y/F")
    t.Branch("data_x", xd, "data_x/F")
    t.Branch("data_y", yd, "data_y/F")

    for i in range(len(A_sim)):
        xs[0], ys[0] = A_sim[i]
        xc[0], yc[0] = A_corr[i]
        if i < len(A_data):
            xd[0], yd[0] = A_data[i]
        t.Fill()

    t.Write()
    f.Close()

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true",help="Run the training step")
    parser.add_argument("--validate",action="store_true",help="Run the validation step")
    args = parser.parse_args()
    
    toy_sim  = ROOT.TFile.Open("toy_simulation.root")
    toy_data = ROOT.TFile.Open("toy_data.root")
    
    source_hists = {}
    target_hists = {}

    # prepare one "validation golden case" to define the convergence for the training
    # with periodic validation and early stopping
    # x=y=a=b=3 for example
    xV=3; yV=3; aV=3; bV=3
    
    # simulation distributions
    Nx=11
    Ny=11
    for ix in range(Nx):
        for iy in range(Ny):
            if ix==xV and iy==yV: continue # skip the validation case
            source_hists[(ix,iy)] = toy_sim.Get(f"h2_{ix}_{iy}")
    
    # data distributions
    Na=5
    Nb=5
    for ia in range(Na):
        for ib in range(Nb):
            if ia==aV and ib==bV: continue # skip the validation case
            target_hists[(ia,ib)] = toy_data.Get(f"h2_{ia}_{ib}")
    
    dataset = TH2UnpairedTransportDataset(
        source_hists,
        target_hists,
        batch_size=32
    )

    # --- TRAINING ---- #
    if args.train: 

        val_case = {
            "A_sim": sample_from_th2(toy_sim.Get(f"h2_{xV}_{yV}"),n_samples=2000),
            "A_data": sample_from_th2(toy_data.Get(f"h2_{aV}_{bV}"),n_samples=2000),
            "context": torch.tensor(
                [[xV, yV, aV, bV]],
                dtype=torch.float32
            )
        }
        
        loader = DataLoader(dataset, batch_size=None, shuffle=True)
     
        context_encoder = ContextEncoder(input_dim=4, # x, y, alpha, beta
                                         hidden_dim=64,
                                         output_dim=32,
                                         n_layers=2,
                                         dropout=0.1
                                         )

        flow = ConditionalFlow(dim=2,
                               n_layers=6,
                               hidden_dim=128,
                               context_dim=32)
     
        # Optimizer
        optimizer = torch.optim.Adam(
            list(flow.parameters()) +
            list(context_encoder.parameters()),
            lr=1e-3
        )
     
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

            losses = train_step(
                flow,
                context_encoder,
                optimizer,
                batch,
                lambda_id=lambda_id,
                sigma_mmd=sigma_mmd
            )
         
            if step % 100 == 0:
                print(
                    f"step {step:5d} | "
                    f"MMD {losses['loss_mmd']:.4f} | "
                    f"ID {losses['loss_id']:.4f}"
                )
         
            # ---- VALIDAZIONE PERIODICA ----
            if step % val_every == 0 and step > 0:
                val_mmd = compute_val_mmd(
                    flow, context_encoder, val_case, sigma_mmd=1.0
                )
         
                print(f"  → val MMD = {val_mmd:.4f}")
         
                if val_mmd < best_val_mmd - min_delta:
                    best_val_mmd = val_mmd
                    best_step = step

                    # Save the output
                    torch.save({
                        "flow_state": flow.state_dict(),
                        "context_state": context_encoder.state_dict(),
                        "flow_config": flow.get_config(),
                        "context_config": context_encoder.get_config(),
                        "best_step": step,
                        "best_val_mmd": best_val_mmd,
                        "sigma_mmd": sigma_mmd,
                        "lambda_id": lambda_id,
                    }, "best_model.pt")
         
                    print(f"  ✓ new best model at step {step}")
         
                elif step - best_step > patience:
                    print(
                        f"Early stopping at step {step} "
                        f"(best step {best_step}, val MMD {best_val_mmd:.4f})"
                    )
                    break
     
        
    # ---  VALIDATION --- #
    elif args.validate:

        # --- CONFIG --- #
        checkpoint_path = "best_model.pt"
        device = "cpu"

        # --- CARICAMENTO MODELLI --- #
        flow, context_encoder, meta = load_model(checkpoint_path, device=device)
        print("Modello caricato!")
        print("Step migliore:", meta.get("best_step"))
        print("Val MMD:", meta.get("best_val_mmd"))

        # --- ESEMPIO: generiamo un caso di validazione per una coppia (x,y) ---
        # seleziona uno xy di validazione
        # Nominal simulation, e.g. "central pair" of sim parameters x0,y0
        # observed environmental parameters (example within the training range) a0,b0
        x0=5;y0=5;a0=2;b0=1
        A_sim = sample_from_th2(source_hists[(x0,y0)],n_samples=50000)
        A_data = sample_from_th2(target_hists[a0,b0],n_samples=50000)
        
        # seleziona un caso di context per il flow
        context_values = [x0, y0, a0, b0]
        context_tensor = torch.tensor([context_values]*len(A_sim), dtype=torch.float32)
        cond = context_encoder(context_tensor)

        # --- APPLICA FLOW PER LA VALIDAZIONE --- #
        print ("EVALUATE FLOW")
        A_corr = flow(A_sim.detach().clone(), cond)
        print("FLOW done")

        # --- CREAZIONE VALIDATOR --- #
        validator = TH2FlowValidation(A_sim=A_sim,
                                      A_corr=A_corr,
                                      A_data=A_data,
                                      xmin=-3,xmax=7,
                                      ymin=-5,ymax=5)

        # --- PRODUZIONE PLOT MULTIPAGINA PDF --- #
        pdf_name = "validation_multipage.pdf"
        validator.plot_validation(pdf_name)
        print(f"Validazione completata. File PDF generato: {pdf_name}")

        # --- SALVA IL ROOT FILE CON IL TREE --- #
        save_arrays_root("trained_tree_valid.root", A_sim, A_corr, A_data)
        
        
    else:
        print("Specify at least --train or --validate")
        exit(0)
        
