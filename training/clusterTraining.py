import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from data_reading.clusterDataset import ConditionalClusterDataset
from data_reading.read_data_2D import make_cygno_collate_fn

class ConditionEncoder(nn.Module):

    def __init__(self, emb_dim=64):
        super().__init__()

        self.sim_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim)
        )

        self.data_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, emb_dim)
        )

    def forward(self, sim_cond, data_cond):
        # Generiamo l'embedding della SIM
        e_sim = self.sim_encoder(sim_cond)

        # Controllo di identità geometrico: se le shape dell'input coincidono 
        # significa che stiamo passando sim_cond anche nel secondo argomento.
        if sim_cond.shape[-1] == data_cond.shape[-1] and torch.equal(sim_cond, data_cond):
            # Usiamo il sim_encoder anche per il target, dato che è un passo di identità SIM->SIM
            e_data = self.sim_encoder(data_cond)
        else:
            # Flusso standard di trasporto verso i DATA reali
            e_data = self.data_encoder(data_cond)

        return e_sim, e_data


class ClusterEncoder(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                1, 32,
                4, 2, 1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32, 64,
                4, 2, 1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64, 128,
                4, 2, 1
            ),
            nn.ReLU(),

            nn.Conv2d(
                128, 256,
                4, 2, 1
            ),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc = nn.Linear(
            256 * 4 * 4,
            latent_dim
        )

    def forward(self,x):
        h = self.net(x)
        # Forziamo h ad avere una scala sana (media=0, std=1) lungo la dimensione latente
        h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-6)
        return self.fc(h)


class ClusterDecoder(nn.Module):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        self.fc = nn.Linear(
            latent_dim,
            256 * 4 * 4
        )

        self.net = nn.Sequential(

            nn.ConvTranspose2d(
                256, 128,
                4, 2, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128, 64,
                4, 2, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64, 32,
                4, 2, 1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1,
                4, 2, 1
            )
        )

    def forward(
        self,
        h
    ):

        x = self.fc(h)

        x = x.view(
            -1,
            256,
            4,
            4
        )

        return self.net(x)


class DifferentialTransport(nn.Module):

    def __init__(
        self,
        latent_dim=128,
        cond_dim=64
    ):

        super().__init__()

        # IMPORTANT: Ora l'input è latent_dim + (2 * cond_dim) -> 128 + 128 = 256
        self.net = nn.Sequential(
            nn.Linear(latent_dim + (2 * cond_dim), 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )

        # ---------------------------------------
        # IMPORTANT: learnable residual scale
        # starts at ZERO → identity at init (EDM ho messo 0.01 per non partire dall'identita' e non farlo rimanere bloccato)
        # ---------------------------------------
        self.gamma = nn.Parameter(torch.tensor(0.01))

        # ---------------------------------------
        # stable init
        # ---------------------------------------
        self._init_weights()

    def _init_weights(self):

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # final layer small init (NOT zero!)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

        
    def forward(self, h, cond_total):
        # cond_total contiene [e_sim, e_data] già concatenati a monte
        x = torch.cat([h, cond_total], dim=-1)

        raw_delta = self.net(x)

        # ---------------------------------------
        # normalize update scale
        # ---------------------------------------
        #raw_delta = raw_delta / (raw_delta.std(dim=-1, keepdim=True) + 1e-6)

        # ---------------------------------------
        # controlled residual update
        # ---------------------------------------
        delta_h = self.gamma * raw_delta

        return delta_h


class CygnoTransportModel(nn.Module):

    def __init__(self, latent_dim=128, cond_dim=64, num_scalars=4):
        super().__init__()

        self.encoder = ClusterEncoder(latent_dim)
        self.cond_encoder = ConditionEncoder()
        
        # Passiamo implicitamente cond_dim=64 (il default del ConditionEncoder)
        self.transport = DifferentialTransport(latent_dim, cond_dim=64)
        self.decoder = ClusterDecoder(latent_dim)

        # per correggere gli scalari basandosi sul contesto ambientale
        # (cond_dim * 2 perché concateniamo l'embedding di SIM e DATA)
        self.scalar_transport = nn.Sequential(
            nn.Linear(cond_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_scalars)
        )
        

    def forward(self, sim_img, sim_cond, data_cond, sim_scalars):
        # 1. Immagini e Spazio latente
        h = self.encoder(sim_img)
        e_sim, e_data = self.cond_encoder(sim_cond, data_cond)
        cond_totale = torch.cat([e_sim, e_data], dim=-1)

        delta_h = self.transport(h, cond_totale)
        h_corr = h + delta_h
        pred_img = self.decoder(h_corr)

        # 2. Correzione degli scalari pre-calcolati (Residual Transport)
        # sim_scalars ha forma [B*N, num_scalars]
        delta_scalars = self.scalar_transport(cond_totale)
        pred_scalars = sim_scalars + delta_scalars

        return {
            "pred": pred_img,
            "pred_scalars": pred_scalars,
            "latent": h,
            "delta_h": delta_h
        }
    

def forward_test(inputfile):

    dataset = ConditionalClusterDataset(
        pkl_file=inputfile,
        n_clusters=32
    )
    
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=make_cygno_collate_fn(dataset)
    )

    batch = next(iter(loader))

    model = CygnoTransportModel()

    x = batch["sim_images"]

    B, N, H, W = x.shape

    x = x.reshape(
        B * N,
        1,
        64,
        64
    )
    
    sim_cond = (
        batch["sim_cond"]
        .repeat_interleave(
            N,
            dim=0
        )
    )
    
    data_cond = (
        batch["data_cond"]
        .repeat_interleave(
            N,
            dim=0
        )
    )

    out = model(
        x,
        sim_cond,
        data_cond
    )

    print(
        out["pred"].shape
    )

    print(
        out["delta_h"]
        .norm(dim=-1)
        .mean()
    )



# below training workflow
# pickle
#  ↓
# ConditionalClusterDataset
#  ↓
# DataLoader
#  ↓
# model
#  ↓
# training loop
#  ↓
# checkpoint
#  ↓
# validation / visual test

from torch.utils.data import DataLoader


def build_dataloader(
        inputfile,
        batch_size=4,
        n_clusters=32,
        shuffle=True):

    dataset = ConditionalClusterDataset(
        pkl_file=inputfile,
        n_clusters=n_clusters
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=make_cygno_collate_fn(dataset)
    )

    return dataset, loader


def compute_mmd_rbf(X, Y):
    B_X = X.size(0)
    B_Y = Y.size(0)
    
    XX = torch.cdist(X, X, p=2).pow(2)
    YY = torch.cdist(Y, Y, p=2).pow(2)
    XY = torch.cdist(X, Y, p=2).pow(2)
    
    with torch.no_grad():
        median_dist = torch.median(XY)
        gamma = 1.0 / (2.0 * median_dist + 1e-8)
        gamma = torch.clamp(gamma, min=1e-3, max=1e6)
    
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)
    
    mmd = K_XX.sum() / (B_X * (B_X - 1) + 1e-6) + K_YY.sum() / (B_Y * (B_Y - 1) + 1e-6) - 2 * K_XY.sum() / (B_X * B_Y + 1e-6)
    return mmd


def total_variation(x):

    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]

    return dx.abs().mean() + dy.abs().mean()


def get_laplacian_kernel(device, dtype):
    k = torch.tensor([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], device=device, dtype=dtype)
    return k

def laplacian_smoothness(x):
    """
    x: [B, C, H, W]  (C = n_clusters)
    """

    B, C, H, W = x.shape

    kernel = get_laplacian_kernel(x.device, x.dtype)

    # [C, 1, 3, 3] -> depthwise
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    lap = F.conv2d(
        x,
        kernel,
        padding=1,
        groups=C
    )

    # smoothness scalar
    return lap.pow(2).mean()

def extract_profiles(x):
    # x ha shape [BatchTotale, 64, 64]
    profilo_x = x.sum(dim=-2) # Somma lungo le righe -> [BatchTotale, 64]
    profilo_y = x.sum(dim=-1) # Somma lungo le colonne -> [BatchTotale, 64]
    # Concateniamo i due profili per ottenere un vettore di feature da 128 elementi
    return torch.cat([profilo_x, profilo_y], dim=-1)


# === COMPLETE LOSS FUNCTION ===
# A) distribution matching
# B) physics loss
# C) scalar auxiliary loss
# D) latent regularization
def compute_cygno_loss(
    pred,
    data,
    pred_scalars,
    target_scalars,
    delta_h,
    pred_identity=None,
    sim_images=None
):

    # 1. Portiamo subito i tensori in formato Flat [StepTotati, H, W] 
    # per non fare confusione con B e N nelle loss statistiche
    # pred shape nativa: [B, N, H, W] -> diventerà [B*N, H, W]
    pred_flat = pred.view(-1, pred.shape[-2], pred.shape[-1])
    data_flat = data.view(-1, data.shape[-2], data.shape[-1])

    # 2. Pulizia dei dati reali dai pixel negativi (Noise Clamping)
    # Importante: i dati simulati sono già positivi (grazie a ELU+1), 
    # ma i dati reali hanno fluttuazioni negative del piedistallo.
    data_flat = torch.clamp(data_flat, min=0.0)

    # 3. Normalizzazione a densità probabilistica spaziale (Somma = 1 per ogni singolo cluster)
    pred_n = pred_flat / (pred_flat.sum(dim=(-1, -2), keepdim=True) + 1e-8)
    data_n = data_flat / (data_flat.sum(dim=(-1, -2), keepdim=True) + 1e-8)

    # print("SHAPE CHECK:")
    # print("pred_n shape:", pred_n.shape)  # Deve essere [B*N, 1, 64, 64] o [B, N, 64, 64]
    # print("data_n shape:", data_n.shape)  # Deve essere identica a pred_n
    # print("RANGE CHECK (Primo elemento del batch):")
    # print(f"PRED_N -> Min: {pred_n[0].min().item():.4f} | Max: {pred_n[0].max().item():.4f} | Sum: {pred_n[0].sum().item():.4f}")
    # print(f"DATA_N -> Min: {data_n[0].min().item():.4f} | Max: {data_n[0].max().item():.4f} | Sum: {data_n[0].sum().item():.4f}")

    # Estraiamo i profili proiettati (Shape finale: [B*N, 128])
    pred_n_feat = extract_profiles(pred_n)
    data_n_feat = extract_profiles(data_n)

    # Calcoliamo la loss sulle forme geometriche dei profili
    L_mmd = compute_mmd_rbf(pred_n_feat, data_n_feat)
    
    # --------------------------------
    # physics constraints
    # --------------------------------
    pred_integral = pred.sum(
        dim=(-1, -2)
    )

    data_integral = data.sum(
        dim=(-1, -2)
    )

    L_integral = (
        pred_integral
        -
        data_integral
    ).pow(2).mean()

    # normalize to the number of elements
    L_integral = L_integral / pred.numel()

    pred_rms = torch.sqrt(
        (pred ** 2).mean(dim=(-1,-2))
    )

    data_rms = torch.sqrt(
        (data ** 2).mean(dim=(-1,-2))
    )
    
    L_rms = (
        pred_rms
        -
        data_rms
    ).pow(2).mean()

    # --------------------------------
    # auxiliary scalar supervision
    # --------------------------------
    # Calcoliamo medie e deviazioni standard lungo il batch (dim=0)
    mean_pred = pred_scalars.mean(dim=0)
    mean_tgt  = target_scalars.mean(dim=0)
    
    std_pred  = pred_scalars.std(dim=0)
    std_tgt   = target_scalars.std(dim=0)

    # Normalizziamo l'MSE dividendo per il target reale (più epsilon anti-zero).
    # In questo modo un errore di 1000 su un target di 20000 peserà solo (1000/20000)^2 = 0.0025,
    # neutralizzando l'esplosione dei gradienti dovuta all'unità di misura sCMOS.
    L_scalars_mean = F.mse_loss(
        mean_pred / (mean_tgt + 1e-5), 
        mean_tgt / (mean_tgt + 1e-5)
    )
    
    L_scalars_std = F.mse_loss(
        std_pred / (std_tgt + 1e-5), 
        std_tgt / (std_tgt + 1e-5)
    )
    
    L_aux = L_scalars_mean + L_scalars_std

    # ------------------------------------------------------------
    # SANITY CHECK DELLE VARIABILI SCALARI
    # ------------------------------------------------------------
    with torch.no_grad():
        # Portiamo i tensori in formato flat per analizzarli
        # pred_scalars e target_scalars hanno shape [B, N, 4] -> passiamo a [B*N, 4]
        p_scal_flat = pred_scalars.view(-1, pred_scalars.shape[-1])
        t_scal_flat = target_scalars.view(-1, target_scalars.shape[-1])
        
        print("\n=== SCALARS SANITY CHECK ===")
        # Cicliamo sulle 4 proprietà fisiche del cluster
        for idx in range(p_scal_flat.shape[-1]):
            p_mean = p_scal_flat[:, idx].mean().item()
            p_std  = p_scal_flat[:, idx].std().item()
            t_mean = t_scal_flat[:, idx].mean().item()
            t_std  = t_scal_flat[:, idx].std().item()
            
            print(f"Scalare [{idx}]:")
            print(f"  -> PREDICTED : Media = {p_mean:12.4f} | Std = {p_std:12.4f}")
            print(f"  -> EXPERIMENTAL: Media = {t_mean:12.4f} | Std = {t_std:12.4f}")
        print("============================\n")
        
    # --------------------------------
    # latent near-identity
    # --------------------------------
    L_transport = (
        delta_h.pow(2)
    ).mean()
    
    # ------------------------------------------------------------
    # total variation loss and smoothness (to reduce pixels jumps)
    # ------------------------------------------------------------
    L_tv = total_variation(pred)
    L_lap = laplacian_smoothness(pred)

    # ------------------------------------------------------------
    # Nuova Loss di Identità (Autoencoder)
    # ------------------------------------------------------------
    L_identity = 0.0
    if pred_identity is not None and sim_images is not None:
        # Applichiamo il softplus anche qui per consistenza con l'output intensità
        pred_id_clamped = F.elu(pred_identity) + 1
        # Semplice MSE a livello di pixel tra l'input SIM e la sua ricostruzione
        L_identity = F.mse_loss(pred_id_clamped, sim_images)
        
    # --------------------------------
    # final weighted loss
    # --------------------------------
    loss = (

        1.0 * L_mmd
        +
        0.1 * L_integral
        +
        0.1 * L_rms
        +
        0.001 * L_aux
        +
        0.0 * L_transport
        +
        0.1 * L_tv
        +
        0.1 * L_lap
        +
        2.0 * L_identity
    )

    loss_dict = {
        "total": loss.item(),
        "mmd": L_mmd.item(),
        "integral": L_integral.item(),
        "rms": L_rms.item(),
        "aux": L_aux.item(),
        "transport": L_transport.item(),
        "totvar": L_tv.item(),
        "laplace": L_tv.item(),
        "identity": L_identity.item() if isinstance(L_identity, torch.Tensor) else 0.0
    }

    return loss, loss_dict



# === TRAINING EPOCH ===
def train_epoch(
        model,
        loader,
        optimizer,
        device="mps",
        max_batches=None):

    model.train()

    epoch_stats = {
        "loss": [],
        "mmd": [],
        "integral": [],
        "rms": [],
        "aux": [],
        "transport": [],
        "totvar": [],
        "laplace": [],
        "delta_h": [],
        "identity": []
    }
    

    print(f"\n\tNumber of batches in this epoch: {len(loader)}")

    for ibatch, batch in enumerate(loader):
        
        if (max_batches is not None
            and ibatch >= max_batches
            ):
            break

        if ibatch % 10 == 0:
            print (f"\t\t  running ibatch {ibatch}...")
    
        sim_images = batch["sim_images"].to(device)
        data_images = batch["data_images"].to(device)

        sim_cond = batch["sim_cond"].to(device)
        data_cond = batch["data_cond"].to(device)

        sim_scalars_raw = batch["sim_scalars"].to(device)
        data_scalars_raw = batch["data_scalars"].to(device)

        B, N, H, W = sim_images.shape
        num_scal = sim_scalars_raw.shape[-1] # Numero di scalari totali ordinati alfabeticamente

        # Appiatto le dimensioni spaziali e dei micro-batch per la rete
        sim_flat = sim_images.view(B * N, 1, H, W)
        sim_cond_flat = (sim_cond.repeat_interleave(N, dim=0))
        data_cond_flat = (data_cond.repeat_interleave(N, dim=0))

        sim_scalars_flat = sim_scalars_raw.view(B * N, num_scal)
        data_scalars_flat = data_scalars_raw.view(B * N, num_scal)
        
        out = model(sim_flat, sim_cond_flat, data_cond_flat, sim_scalars_flat)

        pred = out["pred"]                  # Shape: [B*N, 1, 64, 64]
        pred_scalars = out["pred_scalars"]  # Shape: [B*N, num_scal]
        target_scalars = data_scalars_flat  # Shape: [B*N, num_scal]

        # to force DeltaH=0 for same input/output conditions (sim_cond)
        out_identity = model(sim_flat, sim_cond_flat, sim_cond_flat, sim_scalars_flat)

        pred_identity = out_identity["pred"]
        
        #print(f"\t\t\t ---> pred sum: {pred.sum().item()}, data sum: {data.sum().item()}")
        
        loss, info = (
            compute_cygno_loss(
                pred,
                data_images,
                pred_scalars,
                target_scalars,
                delta_h=out["delta_h"],
                pred_identity=pred_identity,
                sim_images=sim_images
            )
        )
        
        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        # ------------------------
        # accumulate stats
        # ------------------------

        epoch_stats[
            "loss"
        ].append(
            info["total"]
        )

        epoch_stats[
            "mmd"
        ].append(
            info["mmd"]
        )

        epoch_stats[
            "integral"
        ].append(
            info["integral"]
        )

        epoch_stats[
            "rms"
        ].append(
            info["rms"]
        )

        epoch_stats[
            "aux"
        ].append(
            info["aux"]
        )

        epoch_stats[
            "transport"
        ].append(
            info["transport"]
        )

        epoch_stats[
            "totvar"
        ].append(
            info["totvar"]
        )

        epoch_stats[
            "laplace"
        ].append(
            info["laplace"]
        )

        epoch_stats[
            "identity"
        ].append(
            info["identity"]
        )

        epoch_stats[
            "delta_h"
        ].append(
            out["delta_h"]
            .norm()
            .item()
        )

    # ------------------------
    # epoch average
    # ------------------------
    epoch_stats = {
        k: np.mean(v)
        for k, v in
        epoch_stats.items()
    }

    return epoch_stats


# === FULL TRAINING ===
def train_model(inputfile,outputfile,epochs=10):
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    _, loader = build_dataloader(
        inputfile
    )

    # to get the # of scalars
    sample_batch = next(iter(loader))
    n_scalar_features = sample_batch["sim_scalars"].shape[-1] 
    print(f"Found {n_scalar_features} scalars in the dataset. Initialize the model now...")
    
    model = (
        CygnoTransportModel(latent_dim=128, 
                            cond_dim=64, 
                            num_scalars=n_scalar_features
                            ).to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    train_history = {
        "loss": [],
        "mmd": [],
        "aux": [],
        "integral": [],
        "rms": [],
        "transport": [],
        "totvar": [],
        "laplace": [],
        "delta_h": []
    }
    
    print(f"Initialized the model. Now start the training on the device: {device}")
    for epoch in range(epochs):

        print(f"\t|Start epoch n. {epoch}...")

        stats = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            max_batches=100
        )

        for k in train_history:
            train_history[k].append(
                stats[k]
            )

        print(f"epoch {epoch}")

        for k, v in stats.items():
            print(
                f"{k}: "
                f"{v:.4f}"
            )
                
    torch.save(
        model.state_dict(),
        outputfile
    )

    return (model,
            train_history)


def plot_training_history(train_history):

    import matplotlib.pyplot as plt

    for key in train_history:
        plt.figure(
            figsize=(6,4)
        )
        plt.plot(
            train_history[key]
        )
        plt.xlabel(
            "epoch"
        )
        plt.ylabel(
            key
        )
        plt.title(
            key
        )
        plt.grid()

    plt.show()

# === test of the training ===
def test_training(
    model_or_path,
    inputfile,
    device=None):

    # -----------------------
    # device
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
    if isinstance(
        model_or_path,
        str
    ):

        print(
            f"\nLoading model "
            f"from:\n"
            f"{model_or_path}"
        )

        model = (
            CygnoTransportModel()
            .to(device)
        )

        state = torch.load(
            model_or_path,
            map_location=device
        )

        model.load_state_dict(
            state
        )

    else:

        model = model_or_path.to(
            device
        )

    model.eval()

    # -----------------------
    # dataloader
    # -----------------------
    _, loader = build_dataloader(inputfile, batch_size=2) # <--- Metti a 2!
    batch = next(iter(loader))

    sim = batch[
        "sim_images"
    ].to(device)

    data = batch[
        "data_images"
    ].to(device)

    sim_cond = batch[
        "sim_cond"
    ].to(device)

    data_cond = batch[
        "data_cond"
    ].to(device)

    B, N, H, W = sim.shape

    sim_flat = sim.view(
        B * N,
        1,
        H,
        W
    )

    sim_cond = (
        sim_cond
        .repeat_interleave(
            N,
            dim=0
        )
    )

    data_cond = (
        data_cond
        .repeat_interleave(
            N,
            dim=0
        )
    )

    with torch.no_grad():

        out = model(
            sim_flat,
            sim_cond,
            data_cond
        )

    pred = out["pred"].view(
        B,
        N,
        H,
        W
    )

    print("DEBUG TRA BATCH DIFFERENTI: ")
    # Confrontiamo l'evento 0 del batch 0 con l'evento 0 del batch 1
    print(pred[0,0].mean(), pred[1,0].mean())
    
    corr = torch.corrcoef(
        torch.stack([
            pred[0,0].flatten(),
            pred[1,0].flatten()
        ])
    )
    print("corcoeff vero:")
    print(corr)
    print("END DEBUG.")
    
    # -----------------------
    # visual test
    # -----------------------
    # -----------------------------------------------------------------
    # NUOVO CODICE PER IL VISUAL TEST (Variazione tra condizioni diverse)
    # -----------------------------------------------------------------
    import matplotlib.pyplot as plt
    
    # Impostiamo il loader del test per avere batch_size = 1
    # Vogliamo raccogliere 10 batch distinti per avere 10 condizioni diverse
    test_sims = []
    test_preds = []
    test_datas = []
    
    model.eval()
    with torch.no_grad():
        for ibatch, batch in enumerate(loader):
            if ibatch >= 10:  # Ci fermiamo quando abbiamo 10 batch diversi
                break
                
            sim = batch["sim_images"].to(device)
            sim_cond = batch["sim_cond"].to(device)
            data_cond = batch["data_cond"].to(device)
            
            B, N, H, W = sim.shape
            sim_flat = sim.view(B * N, 1, H, W)
            
            # Espandiamo le condizioni per il match flat
            sim_cond_flat = sim_cond.repeat_interleave(N, dim=0)
            data_cond_flat = data_cond.repeat_interleave(N, dim=0)
            
            # Forward
            out = model(sim_flat, sim_cond_flat, data_cond_flat)
            pred_flat = F.elu(out["pred"]) + 1.0
            pred = pred_flat.view(B, N, H, W)
            
            # Scegliamo il primo sotto-cluster (idx=0) di questo specifico batch
            test_sims.append(sim[0, 0].cpu())
            test_preds.append(pred[0, 0].cpu())
            test_datas.append(batch["data_images"][0, 0].cpu())

    # Ora disegnamo le 10 righe, ognuna corrispondente a un BATCH differente
    fig, ax = plt.subplots(10, 3, figsize=(9, 20))
    
    for i in range(10):

        # Troviamo il massimo assoluto di intensità per QUESTA specifica riga
        # escludendo 'data' se ha una dinamica completamente fuori scala, 
        # o includendolo per un confronto assoluto.
        vmax = max(
            test_sims[i].max().item(),
            test_preds[i].max().item(),
            test_datas[i].max().item()
        )
        # Se preferisci vedere le shape normalizzate alla loro intensità usa il vmax locale,
        # ma per vedere la scala z reale usiamo questo vmax unico per la riga:
        
        ax[i,0].imshow(test_sims[i], origin="lower", vmin=0, vmax=vmax, cmap='viridis')
        ax[i,0].set_title(f"SIM (Batch {i})") if i==0 else None
        
        ax[i,1].imshow(test_preds[i], origin="lower", vmin=0, vmax=vmax, cmap='viridis')
        ax[i,1].set_title(f"CORRECTED (Batch {i})") if i==0 else None
        
        # Nota: se i DATI reali hanno un guadagno intrinseco totalmente diverso, 
        # conviene lasciargli il suo vmax per studiare la shape, altrimenti mettiamo vmax anche qui
        ax[i,2].imshow(test_datas[i], origin="lower", vmin=0, vmax=vmax, cmap='viridis')
        ax[i,2].set_title(f"DATA (Batch {i})") if i==0 else None
        
    plt.tight_layout()
    plt.show()
    
    
