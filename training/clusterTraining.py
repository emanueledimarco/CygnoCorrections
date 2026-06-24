import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os

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

def compute_physical_scalars_from_image(images, eps=1e-4):
    """
    Versione ultra-blindata che appiatisce preventivamente qualsiasi 
    struttura di batch/canale complessa in un unico asse di cluster distinti.
    """
    device = images.device
    
    # Se ci passano [B, N, H, W] o [N, C, H, W], collassiamo tutto ciò che sta prima di (H, W)
    H, W = images.shape[-2], images.shape[-1]
    imgs = images.view(-1, H, W) # Diventa rigidamente [Tot_Clusters, 64, 64]
    L_batch = imgs.shape[0]      # Il vero numero di cluster totali da elaborare
    
    # Griglia di coordinate centrate (valori da -32 a 31)
    y_indices, x_indices = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) - H // 2,
        torch.arange(W, dtype=torch.float32, device=device) - W // 2,
        indexing="ij"
    )
    
    x_coords = x_indices.unsqueeze(0) # [1, H, W]
    y_coords = y_indices.unsqueeze(0) # [1, H, W]
    
    imgs = torch.clamp(imgs, min=0.0)
    
    # --- A. INTEGRALE ---
    integrals = torch.sum(imgs, dim=[1, 2]) # [L_batch]
    integrals_safe = torch.where(integrals > eps, integrals, torch.tensor(eps, device=device))
    
    # --- B. CENTROIDI ---
    x_c = torch.sum(imgs * x_coords, dim=[1, 2]) / integrals_safe # [L_batch]
    y_c = torch.sum(imgs * y_coords, dim=[1, 2]) / integrals_safe # [L_batch]
    
    x_c_grid = x_c.view(L_batch, 1, 1)
    y_c_grid = y_c.view(L_batch, 1, 1)

    x_centered = x_coords - x_c_grid
    y_centered = y_coords - y_c_grid
    
    # --- C. MOMENTI SECONDI ---
    mu_xx = torch.sum(imgs * (x_centered ** 2), dim=[1, 2]) / integrals_safe
    mu_yy = torch.sum(imgs * (y_centered ** 2), dim=[1, 2]) / integrals_safe
    mu_xy = torch.sum(imgs * (x_centered * y_centered), dim=[1, 2]) / integrals_safe
    
    # --- D. AUTOVALORI PROTETTI ---
    trace = mu_xx + mu_yy
    det = mu_xx * mu_yy - (mu_xy ** 2)
    
    discriminant_arg = torch.clamp(trace**2 - 4 * det, min=0.0)
    discriminant = torch.sqrt(discriminant_arg + eps)
    
    lambda_max = (trace + discriminant) / 2.0
    lambda_min = (trace - discriminant) / 2.0
    
    lengths = 2.0 * torch.sqrt(torch.clamp(lambda_max, min=0.0) + eps)
    widths = 2.0 * torch.sqrt(torch.clamp(lambda_min, min=0.0) + eps)
    
    # Restituisce [L_batch, 3]
    return torch.stack([integrals, lengths, widths], dim=1)

class CygnoTransportModel(nn.Module):

    def __init__(self, latent_dim=128, cond_dim=64):
        super().__init__()

        self.encoder = ClusterEncoder(latent_dim)
        self.cond_encoder = ConditionEncoder()
        
        # Passiamo implicitamente cond_dim=64 (il default del ConditionEncoder)
        self.transport = DifferentialTransport(latent_dim, cond_dim=cond_dim)
        self.decoder = ClusterDecoder(latent_dim)

    def forward(self, sim_img, sim_cond, data_cond, sim_scalars):
        """
        Esegue il trasporto condizionale nello spazio latente geometrico.
        L'argomento sim_scalars viene mantenuto nella firma per retro-compatibilità 
        con le chiamate esterne (se necessario alla rete latente), ma il calcolo degli scalari 
        predetti viene rimosso poiché ora delegato alla funzione di calcolo dai pixel nella loss.
        """
        # 1. Encoding dell'immagine di simulazione nello spazio latente h
        h = self.encoder(sim_img)
        
        # 2. Estrazione degli embedding di contesto (SIM e DATA)
        e_sim, e_data = self.cond_encoder(sim_cond, data_cond)
        cond_totale = torch.cat([e_sim, e_data], dim=-1)

        # 3. Trasporto differenziale condizionale nello spazio latente
        delta_h = self.transport(h, cond_totale)
        h_corr = h + delta_h
        
        # 4. Decoding dell'immagine trasportata finale (PRED)
        pred_img = self.decoder(h_corr)

        return {
            "pred_images": pred_img,  # Usiamo esplicitamente 'pred_images' per coerenza con train_epoch e il test loop
            "latent": h,
            "delta_h": delta_h
        }
    

def forward_test(inputfile):

    dataset = ConditionalClusterDataset(
        pkl_file=inputfile,
        n_clusters=32,
        is_test=True
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
        out["pred_images"].shape
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
        shuffle=True,
        is_test=False,
):

    dataset = ConditionalClusterDataset(
        pkl_file=inputfile,
        n_clusters=n_clusters,
        is_test=is_test
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
    # Spostiamo temporaneamente i due piccoli vettori delle feature su CPU
    device_originale = X.device
    X = X.cpu()
    Y = Y.cpu()
    
    B_X = X.size(0)
    B_Y = Y.size(0)
    
    # Ora cdist gira su CPU dove il backward è perfettamente supportato!
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
    
    # Riportiamo il valore scalare della loss sul device originale (MPS)
    # per sommarlo coerentemente alle altre loss
    return mmd.to(device_originale)


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
def compute_cygno_loss(
    pred,               # Immagine 2D generata dal Flow: [N, 1, 64, 64]
    data,               # Immagine 2D target reale: [N, 1, 64, 64]
    pred_scalars,       # Vettore [N, 3] di [integral, length, width] da pred_images
    target_scalars,     # Vettore [N, 3] di [integral, length, width] da data_images
    delta_h,            # Vettore di spostamento latente del Flow
    pred_identity=None, # Immagine 2D dell'identità: [N, 1, 64, 64]
    sim_images=None     # Immagine 2D di partenza SIM: [N, 1, 64, 64]
):
    # Noise Clamping sui dati reali sCMOS
    data_clamped = torch.clamp(data, min=0.0)

    # Garantiamo che tutte le intensità predette siano strettamente >= 0 (Consistenza di carica)
    pred_clamped = F.elu(pred) + 1.0

    # ------------------------------------------------------------
    # 1. Identity Loss (Autoencoder di stabilità per il trasporto)
    # ------------------------------------------------------------
    L_identity = 0.0
    if pred_identity is not None and sim_images is not None:
        pred_id_clamped = F.elu(pred_identity) + 1.0
        L_identity = F.mse_loss(pred_id_clamped, sim_images)

    # ------------------------------------------------------------
    # 2. MMD Loss sulle forme geometriche proiettate (Profili X e Y)
    # ------------------------------------------------------------
    # Normalizzazione a densità probabilistica spaziale (Somma dei pixel = 1)
    pred_n = pred_clamped / (pred_clamped.sum(dim=(-1, -2), keepdim=True) + 1e-8)
    data_n = data_clamped / (data_clamped.sum(dim=(-1, -2), keepdim=True) + 1e-8)

    pred_n_feat = extract_profiles(pred_n)
    data_n_feat = extract_profiles(data_n)
    L_mmd = compute_mmd_rbf(pred_n_feat, data_n_feat)

    # ------------------------------------------------------------
    # 3. Supervisione Diretta delle Proprietà Fisiche (Stabilizzata in Log)
    # ------------------------------------------------------------
    # pred_scalars e target_scalars: [N, 3] -> [Integrale, Length, Width]
    # Usiamo il logaritmo per schiacciare la scala dinamica sCMOS da lineare a logaritmica
    log_pred_integral = torch.log(pred_scalars[:, 0] + 1e-3)
    log_target_integral = torch.log(target_scalars[:, 0] + 1e-3)
    
    # Loss sull'integrale in scala logaritmica (MSE del log o Smooth L1)
    L_integral = F.smooth_l1_loss(log_pred_integral, log_target_integral)

    # Anche per la width (diffusione), usiamo un errore relativo robusto (L1) invece dell'MSE quadratico
    # per evitare che un cluster largo sballi l'intero batch
    L_width = F.l1_loss(pred_scalars[:, 2] / (target_scalars[:, 2] + 1e-3), 
                        target_scalars[:, 2] / (target_scalars[:, 2] + 1e-3))

    L_aux = L_integral + L_width
        
    # ------------------------------------------------------------
    # 4. Regolarizzazione Latente e Regolarizzazione Spaziale Pixel
    # ------------------------------------------------------------
    L_transport = (delta_h.pow(2)).mean()
    
    # TV e Laplace per levigare la scacchiera artificiale
    L_tv = total_variation(pred_clamped)
    L_lap = laplacian_smoothness(pred_clamped)

    # ------------------------------------------------------------
    # 5. Combinazione Pesata Finale con Nuovi Bilanciamenti
    # ------------------------------------------------------------
    # Scaliamo L_integral per evitare che cannibalizzi i gradienti
    loss = (
        1.0 * L_mmd
        +
        0.1 * L_integral  # Abbassato a 0.1 per dare spazio alle forme 2D
        +
        1.0 * L_width     # Lasciato a 1.0 per forzare la diffusione trasversa
        +
        0.0 * L_transport
        +
        0.2 * L_tv       
        +
        0.2 * L_lap      
        +
        2.0 * L_identity
    )

    loss_dict = {
        "total": loss.item(),
        "mmd": L_mmd.item(),
        "integral": L_integral.item(),
        "rms": L_width.item(), # mappiamo la width nella vecchia voce RMS per non rompere i log dell'epoca
        "aux": L_aux.item(),
        "transport": L_transport.item(),
        "totvar": L_tv.item(),
        "laplace": L_lap.item(),
        "identity": L_identity.item() if isinstance(L_identity, torch.Tensor) else 0.0
    }

    return loss, loss_dict


# === TRAINING EPOCH ===
import numpy as np
import torch

def train_epoch(model, loader, optimizer, device="mps", max_batches=None):

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
        "identity": [],
    }

    print(f"\n\tNumber of batches in this epoch: {len(loader)}")

    for ibatch, batch in enumerate(loader):

        if max_batches is not None and ibatch >= max_batches:
            break

        if ibatch % 10 == 0:
            print(f"\t\t  running ibatch {ibatch}...")

        # Caricamento delle immagini e condizioni [B, N, H, W]
        sim_images = batch["sim_images"].to(device)
        data_images = batch["data_images"].to(device)
        sim_cond = batch["sim_cond"].to(device)
        data_cond = batch["data_cond"].to(device)

        B, N, H, W = sim_images.shape

        # 1. RICALCOLO MASSIVO DEGLI SCALARI DI PARTENZA E TARGET IN DIRETTA DALLE IMMAGINI
        # Appiattiamo temporaneamente in [B*N, 1, H, W] per far lavorare la funzione in parallelo
        sim_images_flat = sim_images.view(B * N, 1, H, W)
        data_images_flat = data_images.view(B * N, 1, H, W)

        sim_scalars_phys_flat = compute_physical_scalars_from_image(sim_images_flat)
        data_scalars_phys_flat = compute_physical_scalars_from_image(data_images_flat)

        # Ripristiniamo la shape originale ad eventi: [B, N, num_scal] dove num_scal = 3
        sim_scalars_phys = sim_scalars_phys_flat.view(B, N, -1)
        data_scalars_phys = data_scalars_phys_flat.view(B, N, -1)

        # Inizializziamo gli accumulatori per questo macro-batch
        loss_batch_accumulata = 0.0
        delta_h_norm_accumulata = 0.0

        info_batch_accumulato = {
            "total": 0.0,
            "mmd": 0.0,
            "integral": 0.0,
            "rms": 0.0,
            "aux": 0.0,
            "transport": 0.0,
            "totvar": 0.0,
            "laplace": 0.0,
            "identity": 0.0,
        }

        optimizer.zero_grad()

        # ============================================================
        # LOOP ISOLATO SUL SINGOLO EVENTO/CONTESTO REALISTICO (b)
        # ============================================================
        for b in range(B):
            # Isolamento dei N cluster del singolo evento b
            s_img = sim_images[b].unsqueeze(1)   # [N, 1, H, W]
            d_img = data_images[b].unsqueeze(1)   # [N, 1, H, W]

            # Espansione locale dei contesti
            s_cond = sim_cond[b].unsqueeze(0).repeat(N, 1)  # [N, cond_dim_sim]
            d_cond = data_cond[b].unsqueeze(0).repeat(N, 1)  # [N, cond_dim_data]

            # Scalari dell'evento estratti dai vettori differenziabili ricalcolati
            s_scal = sim_scalars_phys[b]  # [N, 3] -> [integral, length, width]
            d_scal = data_scalars_phys[b]  # [N, 3] -> [integral, length, width]

            # Forward del modello (Passiamo gli scalari fisici di partenza coerenti)
            out = model(s_img, s_cond, d_cond, s_scal)
            pred_img = out["pred_images"]  # Immagine prodotta dal Flow [N, 1, H, W]

            # 2. CALCOLO IN DIRETTA DEGLI SCALARI DELLA PREDIZIONE (L'unico vero legame differenziabile)
            pred_scalars_phys = compute_physical_scalars_from_image(pred_img) # [N, 3]

            # Forward dell'identità per vincolare la stabilità del network
            out_identity = model(s_img, s_cond, s_cond, s_scal)
            pred_identity_img = out_identity["pred_images"]

            # 3. CHIAMATA A COMPUTE_CYGNO_LOSS CON VARIABILI 100% COERENTI
            loss_evento, info_evento = compute_cygno_loss(
                pred=pred_img,                         # Ora passiamo direttamente il tensore 2D [N, 1, H, W]
                data=d_img,                         # Tensore target 2D [N, 1, H, W]
                pred_scalars=pred_scalars_phys,     # Scalari [N, 3] estratti geometricamente da pred
                target_scalars=d_scal,              # Scalari [N, 3] estratti geometricamente da data
                delta_h=out["delta_h"],
                pred_identity=pred_identity_img,     # Immagine dell'identità per la loss di consistenza
                sim_images=s_img,
            )

            loss_batch_accumulata += loss_evento / B
            delta_h_norm_accumulata += out["delta_h"].norm().item() / B

            for k in info_batch_accumulato.keys():
                info_batch_accumulato[k] += info_evento[k] / B

        # Backward e ottimizzazione sul macro-batch unificato
        loss_batch_accumulata.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulo statistiche
        epoch_stats["loss"].append(info_batch_accumulato["total"])
        epoch_stats["mmd"].append(info_batch_accumulato["mmd"])
        epoch_stats["integral"].append(info_batch_accumulato["integral"])
        epoch_stats["rms"].append(info_batch_accumulato["rms"])
        epoch_stats["aux"].append(info_batch_accumulato["aux"])
        epoch_stats["transport"].append(info_batch_accumulato["transport"])
        epoch_stats["totvar"].append(info_batch_accumulato["totvar"])
        epoch_stats["laplace"].append(info_batch_accumulato["laplace"])
        epoch_stats["identity"].append(info_batch_accumulato["identity"])
        epoch_stats["delta_h"].append(delta_h_norm_accumulata)

    epoch_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
    return epoch_stats




# === FULL TRAINING ===
def train_model(inputfile,outputfile,epochs=10):
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset, loader = build_dataloader(
        inputfile
    )

    # to get the # of scalars
    sample_batch = next(iter(loader))
    
    model = (
        CygnoTransportModel(latent_dim=128, 
                            cond_dim=64, 
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
    output_dir="plot/validation_plots",
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

    os.makedirs(output_dir, exist_ok=True)
    
    # -----------------------
    # dataloader
    # -----------------------
    dataset, loader = build_dataloader(inputfile, batch_size=10, is_test=True)
    batch = next(iter(loader))

    sim = batch["sim_images"].to(device)
    data = batch["data_images"].to(device)

    sim_cond = batch["sim_cond"].to(device)
    data_cond = batch["data_cond"].to(device)

    sim_scalars_raw = batch["sim_scalars"].to(device)
    num_scal = sim_scalars_raw.shape[-1] # Numero di scalari totali ordinati alfabeticamente
    
    B, N, H, W = sim.shape

    sim_flat = sim.view(B * N, 1, H, W)
    data_flat = data.view(B * N, 1, H, W)
    
    sim_cond_flat = (sim_cond.repeat_interleave(N, dim=0))
    data_cond_flat = (data_cond.repeat_interleave(N, dim=0))

    sim_scalars_flat = sim_scalars_raw.view(B * N, num_scal)

    with torch.no_grad():
        out = model(sim_flat, sim_cond_flat, data_cond_flat, sim_scalars_flat)

    # 1. Ricostruisci la struttura a blocchi per le immagini
    pred_clamped_flat = F.elu(out["pred_images"]) + 1.0
    pred_images = pred_clamped_flat.view(B, N, H, W)
    
    # 2. Ricostruisci la struttura a blocchi per gli scalari
    # Da [B*N, num_scalars] a [B, N, num_scalars]
    pred_scalars = compute_physical_scalars_from_image(pred_images) # [N, 3]
    
    print("DEBUG TRA BATCH DIFFERENTI: ")
    # Confrontiamo l'evento 0 del batch 0 con l'evento 0 del batch 1
    print(pred_images[0,0].mean(), pred_images[1,0].mean())
    
    corr = torch.corrcoef(
        torch.stack([
            pred_images[0,0].flatten(),
            pred_images[1,0].flatten()
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
            sim_scalars_raw = batch["sim_scalars"].to(device)
            
            B, N, H, W = sim.shape
            sim_flat = sim.view(B * N, 1, H, W)
            
            # Espandiamo le condizioni per il match flat
            sim_cond_flat = sim_cond.repeat_interleave(N, dim=0)
            data_cond_flat = data_cond.repeat_interleave(N, dim=0)
            sim_scalars_flat = sim_scalars_raw.view(B * N, num_scal)

            # Forward
            out = model(sim_flat, sim_cond_flat, data_cond_flat, sim_scalars_flat)
            pred_clamped_flat = F.elu(out["pred_images"]) + 1.0
            pred_images = pred_clamped_flat.view(B, N, H, W)
            pred_scalars = torch.clamp(compute_physical_scalars_from_image(pred_images), min=0.0).view(B, N, -1)
            
            # Scegliamo il primo sotto-cluster (idx=0) di questo specifico batch
            test_sims.append(sim[0, 0].cpu())
            test_preds.append(pred_images[0, 0].cpu())
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
    cluster_test_path = os.path.join(output_dir, "clusters10_test.png")
    plt.savefig(cluster_test_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\t--> Griglia con 10 clusters salvata con successo in: {cluster_test_path}")

    run_sampled_and_detailed_test(model,loader,device,max_batches=50,output_dir="plot/validation_plots",save_individual_plots=False)

def run_sampled_and_detailed_test(
    model,
    test_loader,
    device,
    max_batches=50,
    num_sim_ctx_to_sample=3,
    num_data_ctx_to_sample=3,
    save_individual_plots=False,
    output_dir="plots",
):
    import matplotlib.pyplot as plt

    """Accumula i dati di test, esegue un campionamento deterministico/selezionato dei contesti per una
    griglia veloce (Csim x Cdata) priva di mixing e, se richiesto, salva grafici 1D separati ad
    alta statistica per ogni combinazione.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_sim_scalars = []
    all_pred_scalars = []
    all_data_scalars = []
    all_sim_cond = []
    all_data_cond = []

    # 1. ACCUMULO VELOCE (TUTTO IN TORCH SU DEVICE)
    print(f"Now accumulating statistics over max {max_batches} batches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            sim_images = batch["sim_images"].to(device)
            data_images = batch["data_images"].to(device)
            B, N, H, W = sim_images.shape
            
            sim_images_flat = sim_images.view(B * N, 1, H, W)
            data_images_flat = data_images.view(B * N, 1, H, W)
            
            sim_cond = batch["sim_cond"].to(device)
            data_cond = batch["data_cond"].to(device)
            
            # Escono come matrici 2D [B*N, 3]
            sim_scalars_raw = compute_physical_scalars_from_image(sim_images_flat)
            data_scalars_raw = compute_physical_scalars_from_image(data_images_flat)

            for b in range(B):
                s_img = sim_images[b].unsqueeze(1) # [N, 1, 64, 64]
                d_img = data_images[b].unsqueeze(1) # [N, 1, 64, 64]

                # Gestione shape cond senza passare da numpy
                if len(sim_cond.shape) == 3:
                    s_c = sim_cond[b]
                    d_c = data_cond[b]
                else:
                    s_c = sim_cond[b].unsqueeze(0).repeat(N, 1)
                    d_c = data_cond[b].unsqueeze(0).repeat(N, 1)

                # CORREZIONE CRITICA: Estraiamo la slice di N cluster per l'evento b mantenendo la shape 2D [N, 3]
                s_scal = sim_scalars_raw[b * N : (b + 1) * N]
                d_scal = data_scalars_raw[b * N : (b + 1) * N]

                # Forward puramente su device
                out = model(s_img, s_c, d_c, s_scal)
                pred_img = out["pred_images"]
                pred_scalars_clamped = torch.clamp(compute_physical_scalars_from_image(pred_img), min=0.0)

                # Accumuliamo i tensori coerentemente tutti come matrici 2D [N, 3] o [N, cond_dim]
                all_sim_scalars.append(s_scal)
                all_pred_scalars.append(pred_scalars_clamped)
                all_data_scalars.append(d_scal)
                all_sim_cond.append(s_c)
                all_data_cond.append(d_c)

    # 2. CONVERSIONE IN NUMPY MASSIVA RIGIDAMENTE 2D [Tot_Clusters, 3]
    sim_sc = torch.cat(all_sim_scalars, dim=0).cpu().numpy()
    pred_sc = torch.cat(all_pred_scalars, dim=0).cpu().numpy()
    data_sc = torch.cat(all_data_scalars, dim=0).cpu().numpy()
    
    sim_co = torch.cat(all_sim_cond, dim=0).cpu().numpy()
    data_co = torch.cat(all_data_cond, dim=0).cpu().numpy()

    # Verifica istantanea di sicurezza nei log (ora stamperà correttamente le colonne)
    print(f"DEBUG MATRICI GRIGLIA -> sim_sc shape: {sim_sc.shape} | pred_sc shape: {pred_sc.shape}")
        
    sim_co_rounded = np.round(sim_co, decimals=4)
    data_co_rounded = np.round(data_co, decimals=4)

    # 3. IDENTIFICAZIONE DELLE COPPIE REALI ED ESTRAZIONE DEGLI ASSI PER IL SUBSET
    coppie_reali = np.unique(
        np.hstack([sim_co_rounded, data_co_rounded]), axis=0
    )
    
    all_unique_sim = np.unique(coppie_reali[:, :3], axis=0)
    all_unique_data = np.unique(coppie_reali[:, 3:], axis=0)

    print(f"Contesti totali nel dataset -> SIM: {len(all_unique_sim)}, DATA: {len(all_unique_data)}")

    # Selezioniamo il subset N x M basandoci su num_sim_ctx_to_sample e num_data_ctx_to_sample
    unique_sim_ctx = all_unique_sim[: min(num_sim_ctx_to_sample, len(all_unique_sim))]
    unique_data_ctx = all_unique_data[: min(num_data_ctx_to_sample, len(all_unique_data))]

    n_rows = len(unique_sim_ctx)
    n_cols = len(unique_data_ctx)
    print(f"Griglia di test campionata impostata a: {n_rows}x{n_cols}")

    # --------------------------------------------------------
    # MODALITÀ A: GENERAZIONE DELLE GRIGLIE PER OGNI VARIABILE
    # --------------------------------------------------------
    coppie_selezionate = coppie_reali[:(num_sim_ctx_to_sample * num_data_ctx_to_sample)]
    
    n_coppie = len(coppie_selezionate)
    if n_coppie == 0:
        print("[ATTENZIONE] Nessuna combinazione trovata nei dati di test accumulati.")
        return

    n_cols = min(3, num_data_ctx_to_sample)
    n_rows = (n_coppie + n_cols - 1) // n_cols

    # Mappatura ordinata delle nostre variabili geometriche reali (lunghezza = 3)
    whitelist_recomputed = ["integral_recomputed", "length_recomputed", "width_recomputed"]
    
    # Determiniamo dinamicamente il numero di scalari dall'array finale per sicurezza
    num_scalars = sim_sc.shape[-1]
    print(f"\t[TEST] Rilevati dinamicamente {num_scalars} scalari fisici da plottare nelle griglie.")
    
    for idx_var in range(num_scalars):
        var_name = whitelist_recomputed[idx_var]
        print(f"\t--> Generazione griglia elegante {n_rows}x{n_cols} per la variabile {var_name}...")
        
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False
        )
        
        axes_flat = axes.flatten()
        idx_coppia = -1

        for idx_coppia, coppia in enumerate(coppie_selezionate):
            ax = axes_flat[idx_coppia]
            
            s_ctx = coppia[:3]
            d_ctx = coppia[3:]  # Contiene [z, P, T, H]

            mask_sim = np.isclose(sim_co_rounded, s_ctx, atol=1e-5).all(axis=1)
            mask_data = np.isclose(data_co_rounded, d_ctx, atol=1e-5).all(axis=1)
            mask = mask_sim & mask_data

            # Ora sim_sc è rigidamente 2D, l'indicizzazione a due coordinate è sicura e corretta
            s_vals = sim_sc[mask, idx_var]
            p_vals = pred_sc[mask, idx_var]
            d_vals = data_sc[mask, idx_var]

            # Se la maschera non seleziona eventi per questa combinazione, saltiamo il plot
            if len(s_vals) == 0 or len(d_vals) == 0:
                ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha="center")
                continue

            # --------------------------------------------------------
            # 1. RANGE DINAMICO BASATO SUI PERCENTILI (Rimuove gli Outlier)
            # --------------------------------------------------------
            all_vals_combined = np.concatenate([s_vals, p_vals, d_vals])
            vmin = np.percentile(all_vals_combined, 1.0)   # Taglia l'1% più basso
            vmax = np.percentile(all_vals_combined, 99.0)  # Taglia l'1% alto
            
            if vmin == vmax:
                vmin, vmax = vmin - 1e-3, vmax + 1e-3

            bins = np.linspace(vmin, vmax, 30)

            # --------------------------------------------------------
            # 2. DISEGNO DEI PLOT (SIM e PRED come Istogrammi)
            # --------------------------------------------------------
            ax.hist(s_vals, bins=bins, alpha=0.5, histtype="step", linewidth=2, label="SIM", color="tab:blue", density=True)
            ax.hist(p_vals, bins=bins, alpha=0.7, histtype="step", linewidth=2, label="PRED", color="tab:orange", density=True)

            # --------------------------------------------------------
            # 3. DATA REALI COME PUNTI CON ERRORE POISSONIANO
            # --------------------------------------------------------
            counts, bin_edges = np.histogram(d_vals, bins=bins)
            counts_density, _ = np.histogram(d_vals, bins=bins, density=True)
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            errors_raw = np.sqrt(counts)
            
            scaling_factor = np.where(counts > 0, counts_density / counts, 0.0)
            errors_density = errors_raw * scaling_factor

            valid_bins = counts > 0
            ax.errorbar(
                bin_centers[valid_bins],
                counts_density[valid_bins],
                yerr=errors_density[valid_bins],
                fmt='o',
                markersize=4,
                color="black",
                ecolor="black",
                capsize=2,
                label="DATA"
            )

            # --------------------------------------------------------
            # 4. TITOLO ELEGANTE CON COMPONENTE CONTESTO
            # --------------------------------------------------------
            titolo_sim = f"SIM: z={int(s_ctx[0])}, α={s_ctx[1]:.2f}, λ={s_ctx[2]:.2f}"
            titolo_data = f"DATA: z={int(d_ctx[0])}, P={int(d_ctx[1])}, T={d_ctx[2]:.1f}, H={d_ctx[3]:.2f}"
            ax.set_title(f"{titolo_sim}\n{titolo_data}", fontsize=8, fontweight="bold")
            
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(loc="upper right", fontsize=8)

            ax.set_ylim(bottom=0.0)

        for idx_retro in range(idx_coppia + 1, len(axes_flat)):
            axes_flat[idx_retro].axis('off')

        plt.tight_layout()
        grid_path = os.path.join(output_dir, f"griglia_ottimizzata_{var_name}.png")
        plt.savefig(grid_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\t--> Griglia salvata con successo in: {grid_path}")


    # 5. MODALITÀ B: PLOT SINGOLI 1D AD ALTA STATISTICA (Per tutte le combinazioni reali)
    if save_individual_plots:
        print("--> Generazione dei plot 1D singoli per ogni contesto...")
        for s_ctx in all_unique_sim:
            for d_ctx in all_unique_data:
                # Applichiamo la tolleranza stretta anche qui per coerenza millimetrica
                mask_sim = np.isclose(sim_co_rounded, s_ctx, atol=1e-5).all(axis=1)
                mask_data = np.isclose(data_co_rounded, d_ctx, atol=1e-5).all(axis=1)
                mask = mask_sim & mask_data

                # Se questa combinazione non ha cluster nel dataset attuale, saltiamo
                if not mask.any():
                    continue

                # Generiamo un file separato per ciascuna variabile di questa combinazione pura
                for var_idx, var_name in enumerate(whitelist):
                    s_vals = sim_sc[mask, var_idx]
                    p_vals = pred_sc[mask, var_idx]
                    d_vals = data_sc[mask, var_idx]

                    plt.figure(figsize=(7, 5))
                    bins = np.linspace(
                        min(s_vals.min(), p_vals.min(), d_vals.min()),
                        max(s_vals.max(), p_vals.max(), d_vals.max()),
                        35,
                    )

                    plt.hist(
                        s_vals,
                        bins=bins,
                        alpha=0.4,
                        label=f"SIM {list(np.round(s_ctx,2))}",
                        color="tab:blue",
                        density=True,
                    )
                    plt.hist(
                        p_vals,
                        bins=bins,
                        histtype="step",
                        linewidth=2.5,
                        label="CORR (Transported)",
                        color="tab:orange",
                        density=True,
                    )
                    plt.hist(
                        d_vals,
                        bins=bins,
                        alpha=0.2,
                        label=f"DATA {list(np.round(d_ctx,2))}",
                        color="tab:green",
                        hatch="//",
                        density=True,
                    )

                    s_str = "_".join([str(x) for x in s_ctx])
                    d_str = "_".join([str(x) for x in d_ctx])

                    plt.title(
                        f"Dettaglio {var_name.upper()}\nSIM:[{s_str}] → DATA:[{d_str}]",
                        fontsize=10,
                        fontweight="bold",
                    )
                    plt.xlabel("Valore Fisico Scalare")
                    plt.ylabel("Densità di Probabilità")
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.legend(loc="upper right")

                    indiv_path = os.path.join(
                        output_dir, f"{var_name}_SIM_{s_str}_DATA_{d_str}.png"
                    )
                    plt.savefig(indiv_path, dpi=150, bbox_inches="tight")
                    plt.close()
