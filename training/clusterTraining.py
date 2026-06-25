import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os

from collections import defaultdict

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

class ResNetBlock(nn.Module):
    """Preserva le alte frequenze geometriche e i dettagli microscopici del 
    cluster senza spezzare il flusso del trasporto ottimale."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Usiamo InstanceNorm per evitare che i bachi di cluster vuoti falsino le statistiche
        self.in1 = nn.InstanceNorm2d(channels)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return F.relu(out + residual)


class FiLMBlob(nn.Module):
    """Modulazione affine basata sul contesto per scalare e traslare le feature map 
    spaziali in base alle condizioni ambientali reali (z, P, T, H)."""
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.fc = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        film_params = self.fc(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # Broadcasting spaziale [B, C, 1, 1]
        gamma = gamma.view(-1, x.size(1), 1, 1) + 1.0
        beta = beta.view(-1, x.size(1), 1, 1)
        return gamma * x + beta


class ResNetClusterEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.init_conv = nn.Conv2d(1, 32, 4, 2, 1)  # -> 32x32
        self.res1 = ResNetBlock(32)
        
        self.layer2 = nn.Conv2d(32, 64, 4, 2, 1)   # -> 16x16
        self.res2 = ResNetBlock(64)
        
        self.layer3 = nn.Conv2d(64, 128, 4, 2, 1)  # -> 8x8
        self.res3 = ResNetBlock(128)
        
        self.layer4 = nn.Conv2d(128, 256, 4, 2, 1) # -> 4x4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.init_conv(x))
        x = self.res1(x)
        x = F.relu(self.layer2(x))
        x = self.res2(x)
        x = F.relu(self.layer3(x))
        x = self.res3(x)
        x = F.relu(self.layer4(x))
        
        h_flat = self.flatten(x)
        h_flat = (h_flat - h_flat.mean(dim=-1, keepdim=True)) / (h_flat.std(dim=-1, keepdim=True) + 1e-6)
        return self.fc(h_flat)


class FiLMResNetDecoder(nn.Module):
    def __init__(self, latent_dim=128, cond_total_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # -> 8x8
        self.film1 = FiLMBlob(cond_total_dim, 128)
        self.res1 = ResNetBlock(128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # -> 16x16
        self.film2 = FiLMBlob(cond_total_dim, 64)
        self.res2 = ResNetBlock(64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # -> 32x32
        self.film3 = FiLMBlob(cond_total_dim, 32)
        self.res3 = ResNetBlock(32)
        
        self.up4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)    # -> 64x64

    def forward(self, h, cond):
        x = self.fc(h).view(-1, 256, 4, 4)
        
        x = self.up1(x)
        x = self.film1(x, cond)
        x = self.res1(x)
        
        x = self.up2(x)
        x = self.film2(x, cond)
        x = self.res2(x)
        
        x = self.up3(x)
        x = self.film3(x, cond)
        x = self.res3(x)
        
        return self.up4(x)


# --- WRAPPER AGGIORNATO PER IL MODELLO PRINCIPALE ---
class CygnoTransportModel(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=64):
        super().__init__()

        self.encoder = ResNetClusterEncoder(latent_dim)
        self.decoder = FiLMResNetDecoder(latent_dim, cond_total_dim=cond_dim*2)
        
        self.cond_encoder = ConditionEncoder()
        self.transport = DifferentialTransport(latent_dim, cond_dim=cond_dim)

    def forward(self, sim_img, sim_cond, data_cond, sim_scalars):
        # 1. Encoding spaziale ad alta fedeltà (ResNet)
        h = self.encoder(sim_img)
        
        # 2. Embedding contestuale condizionato
        e_sim, e_data = self.cond_encoder(sim_cond, data_cond)
        cond_totale = torch.cat([e_sim, e_data], dim=-1)

        # 3. Trasporto differenziale latente 
        delta_h = self.transport(h, cond_totale)
        h_corr = h + delta_h
        
        # 4. Generazione modulata continua (Senza skip-connections "rigide")
        pred_img = self.decoder(h_corr, cond_totale)

        return {
            "pred_images": pred_img,
            "latent": h,
            "delta_h": delta_h
        }
    
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
        # starts at ZERO → identity at init 
        # ---------------------------------------
        self.gamma = nn.Parameter(torch.tensor(1.0))

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
        nn.init.xavier_uniform_(self.net[-1].weight)
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


def compute_centroids(images,eps=1e-4):
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

    return (x_centered,y_centered)

def compute_centroid_loss(sim,pred):
    sim_xc,sim_yc = compute_centroids(sim)
    pred_xc,pred_yc = compute_centroids(pred)
    # Forza il baricentro del cluster trasportato a non deviare dall'originale
    L_centroid = torch.mean((sim_xc - pred_xc)**2 + (sim_yc - pred_yc)**2)
    return L_centroid

def compute_mmd_rbf(X, Y):
    """
    Calcolo MMD Unbiased e Multi-Scala. 
    Gira nativamente su MPS/CUDA senza staccare il grafo computazionale.
    """
    B_X = X.shape[0]
    B_Y = Y.shape[0]
    
    # FORZATURA SHAPE: Appiattiamo eventuali dimensioni extra (come i canali residui)
    # in modo che X e Y siano rigorosamente matrici 2D [Batch, Features]
    X = X.view(B_X, -1)
    Y = Y.view(B_Y, -1)
    
    # 1. Distanza Euclidea al quadrato SENZA radici (Evita i gradienti NaN in 0)
    # Il broadcasting ora produrrà matrici perfettamente 2D: [B, B]
    XX = torch.sum((X.unsqueeze(1) - X.unsqueeze(0)) ** 2, dim=-1)
    YY = torch.sum((Y.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1)
    XY = torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1)
    
    # 2. Kernel RBF Multi-Scala
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    mmd_loss = 0.0
    for alpha in alphas:
        K_XX = torch.exp(- XX / alpha)
        K_YY = torch.exp(- YY / alpha)
        K_XY = torch.exp(- XY / alpha)
        
        # 3. MMD Unbiased 
        # Ora torch.trace lavora su matrici 2D sicure
        K_XX_sum = K_XX.sum() - torch.trace(K_XX)
        K_YY_sum = K_YY.sum() - torch.trace(K_YY)
        
        # Calcolo dei termini normalizzati
        term_XX = K_XX_sum / (B_X * (B_X - 1) + 1e-8)
        term_YY = K_YY_sum / (B_Y * (B_Y - 1) + 1e-8)
        term_XY = 2.0 * K_XY.sum() / (B_X * B_Y + 1e-8)
        
        mmd_loss += (term_XX + term_YY - term_XY)
        
    return mmd_loss

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

def extract_profiles_with_diagonals(img_tensor):
    """
    Estrae i profili X, Y e le due diagonali.
    img_tensor: [B, 1, 64, 64] -> Ritorna feature vector [B, 382]
    """
    B, C, H, W = img_tensor.shape
    img = img_tensor.view(B, H, W)
    
    # Profili cartesiani standard
    prof_x = img.sum(dim=1) # [B, 64]
    prof_y = img.sum(dim=2) # [B, 64]
    
    # Estrazione differenziabile delle diagonali (da offset -63 a +63)
    diags_1 = []
    diags_2 = []
    img_flipped = torch.flip(img, dims=[2]) # Per l'altra diagonale
    
    for offset in range(-H + 1, W):
        # Somma lungo la diagonale principale traslata
        d1 = torch.diagonal(img, offset=offset, dim1=1, dim2=2).sum(dim=1)
        # Somma lungo la diagonale secondaria traslata
        d2 = torch.diagonal(img_flipped, offset=offset, dim1=1, dim2=2).sum(dim=1)
        diags_1.append(d1)
        diags_2.append(d2)
        
    prof_d1 = torch.stack(diags_1, dim=1) # [B, 127]
    prof_d2 = torch.stack(diags_2, dim=1) # [B, 127]
    
    # Concateniamo tutto in un unico vettore morfologico che non perdona i rombi
    return torch.cat([prof_x, prof_y, prof_d1, prof_d2], dim=1)

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

    # # ------------------------------------------------------------
    # # 2. MMD Loss sulle forme geometriche proiettate (Profili X e Y) *normalizzati*
    # # ------------------------------------------------------------
    # Normalizzazione a densità probabilistica spaziale (Somma dei pixel = 1)
    pred_n = pred_clamped / (pred_clamped.sum(dim=(-1, -2), keepdim=True) + 1e-8)
    data_n = data_clamped / (data_clamped.sum(dim=(-1, -2), keepdim=True) + 1e-8)

    pred_n_feat = extract_profiles_with_diagonals(pred_n)
    data_n_feat = extract_profiles_with_diagonals(data_n)
    L_mmd_shape_1d = compute_mmd_rbf(pred_n_feat, data_n_feat)

    # ------------------------------------------------------------
    # 3. Supervisione Diretta delle Proprietà Fisiche *con scala assoluta* (Stabilizzata in Log) 
    # ------------------------------------------------------------
    # 1. Estraiamo le feature fisiche riscaldate (Log per integrale, divisione per le dimensioni stimate)
    # pred_scalars e target_scalars (o data_scalars) hanno shape [Batch, 3] -> [Integrale, Length, Width]
    pred_physics_feat = torch.stack([
        torch.log10(pred_scalars[:, 0] + 1.0),   # Log-Integrale (gestisce ordini di grandezza da 2000 a 50000)
        pred_scalars[:, 1] / 10.0,               # Length riscaldata
        pred_scalars[:, 2] / 10.0                # Width riscaldata
    ], dim=1)

    target_physics_feat = torch.stack([
        torch.log10(target_scalars[:, 0] + 1.0),
    target_scalars[:, 1] / 10.0,
    target_scalars[:, 2] / 10.0
    ], dim=1)

    # 2. Sostituiamo le loss punto a punto con la MMD sulle distribuzioni!
    # Questo non accoppia i cluster uno a uno, ma allinea gli istogrammi globali (media, varianza e code)
    L_mmd_physics = compute_mmd_rbf(pred_physics_feat, target_physics_feat)

    # Puoi mantenere una componente L_integral globale (sulla media del batch) 
    # solo per dare una spinta iniziale forte sull'ordine di grandezza, se necessario:
    L_integral = F.mse_loss(pred_physics_feat[:, 0].mean(), target_physics_feat[:, 0].mean())

    # ------------------------------------------------------------
    # 4. Regolarizzazione Latente e Regolarizzazione Spaziale Pixel
    # ------------------------------------------------------------
    L_transport = (delta_h.pow(2)).mean()
    
    # TV e Laplace per levigare la scacchiera artificiale
    L_tv = total_variation(pred_clamped)
    L_lap = laplacian_smoothness(pred_clamped)

    # Forza il baricentro del cluster trasportato a non deviare dall'originale
    L_centroid = compute_centroid_loss(sim_images,pred)
    
    # ------------------------------------------------------------
    # 5. Combinazione Pesata Finale con Nuovi Bilanciamenti
    # ------------------------------------------------------------
    # Scaliamo L_integral per evitare che cannibalizzi i gradienti
    gamma_mmd_shape_1d = 300.0
    gamma_mmd_physics = 150.0
    gamma_integral = 50.
    gamma_transport = 0.1
    gamma_tv = 0.05
    gamma_lap = 0.05
    gamma_centroid = 3.0
    gamma_identity = 0.0
    
    loss = (
        gamma_mmd_shape_1d * L_mmd_shape_1d
        +
        gamma_mmd_physics * L_mmd_physics
        +
        gamma_integral * L_integral
        +
        gamma_transport * L_transport
        +
        gamma_tv * L_tv       
        +
        gamma_lap * L_lap
        +
        gamma_centroid * L_centroid
        +
        gamma_identity * L_identity
    )

    loss_dict = {
        "total": loss.item(),
        "mmd_shape_1d": gamma_mmd_shape_1d * L_mmd_shape_1d.item(),
        "mmd_physics": gamma_mmd_physics * L_mmd_physics.item(),
        "integral": gamma_integral * L_integral.item(),
        "transport": gamma_transport * L_transport.item(),
        "totvar": gamma_tv * L_tv.item(),
        "laplace": gamma_lap * L_lap.item(),
        "centroid": gamma_centroid * L_centroid.item(),
        "identity": gamma_identity * L_identity.item() if isinstance(L_identity, torch.Tensor) else 0.0
    }

    return loss, loss_dict


# === TRAINING EPOCH ===
import numpy as np
import torch

def train_epoch(model, loader, optimizer, device="mps", max_batches=None):

    model.train()

    epoch_stats = {
        "loss": [],
        "mmd_shape_1d": [],
        "mmd_physics": [],
        "integral": [],
        "transport": [],
        "totvar": [],
        "laplace": [],
        "centroid": [],
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
            "mmd_shape_1d": 0.0,
            "mmd_physics": 0.0,
            "integral": 0.0,
            "transport": 0.0,
            "totvar": 0.0,
            "laplace": 0.0,
            "centroid": 0.0,
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
        epoch_stats["mmd_shape_1d"].append(info_batch_accumulato["mmd_shape_1d"])
        epoch_stats["mmd_physics"].append(info_batch_accumulato["mmd_physics"])
        epoch_stats["integral"].append(info_batch_accumulato["integral"])
        epoch_stats["transport"].append(info_batch_accumulato["transport"])
        epoch_stats["totvar"].append(info_batch_accumulato["totvar"])
        epoch_stats["laplace"].append(info_batch_accumulato["laplace"])
        epoch_stats["centroid"].append(info_batch_accumulato["centroid"])
        epoch_stats["identity"].append(info_batch_accumulato["identity"])
        epoch_stats["delta_h"].append(delta_h_norm_accumulata)

    epoch_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
    return epoch_stats




# === FULL TRAINING ===
def train_model(inputfile,outputfile,epochs=5):
    
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
        "mmd_shape_1d": [],
        "mmd_physics": [],
        "integral": [],
        "transport": [],
        "totvar": [],
        "laplace": [],
        "centroid": [],
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

        plt.tight_layout()
        nameplot = f"training_history_loss_{key}.png"
        plt.savefig(nameplot, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\t--> Saved the loss history plot in: {nameplot}")

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
    import matplotlib.colors as mcolors
    
    # Bianco per il minimo, poi rainbow
    base = plt.get_cmap("rainbow")
    colors = np.vstack([
        np.array([[1, 1, 1, 1]]),      # bianco
        base(np.linspace(0, 1, 255))
    ])

    cmap = mcolors.ListedColormap(colors)
    
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
        
        ax[i,0].imshow(test_sims[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap)
        ax[i,0].set_title(f"SIM (Batch {i})") if i==0 else None
        
        ax[i,1].imshow(test_preds[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap)
        ax[i,1].set_title(f"CORRECTED (Batch {i})") if i==0 else None
        
        # Nota: se i DATI reali hanno un guadagno intrinseco totalmente diverso, 
        # conviene lasciargli il suo vmax per studiare la shape, altrimenti mettiamo vmax anche qui
        ax[i,2].imshow(test_datas[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap)
        ax[i,2].set_title(f"DATA (Batch {i})") if i==0 else None
        
    plt.tight_layout()
    cluster_test_path = os.path.join(output_dir, "clusters10_test.png")
    plt.savefig(cluster_test_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\t--> Griglia con 10 clusters salvata con successo in: {cluster_test_path}")

    run_sampled_and_detailed_test(model,loader,device,max_batches=100,output_dir="plot/validation_plots",save_individual_plots=False)

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

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # 1. INIZIALIZZAZIONE STRUTTURE DATI
    all_sim_scalars = []
    all_pred_scalars = []
    all_data_scalars = []
    
    # AGGIUNGI QUESTA RIGA: inizializzazione del dizionario prima di ogni altra cosa
    from collections import defaultdict
    pair_dict = defaultdict(list)

    # 1. ACCUMULO VELOCE
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
            
            sim_scalars_raw = compute_physical_scalars_from_image(sim_images_flat)
            data_scalars_raw = compute_physical_scalars_from_image(data_images_flat)

            for b in range(B):
                s_img = sim_images[b].unsqueeze(1)
                d_img = data_images[b].unsqueeze(1)

                s_c = sim_cond[b] if len(sim_cond.shape) == 3 else sim_cond[b].unsqueeze(0).repeat(N, 1)
                d_c = data_cond[b] if len(data_cond.shape) == 3 else data_cond[b].unsqueeze(0).repeat(N, 1)

                s_scal = sim_scalars_raw[b * N : (b + 1) * N]
                d_scal = data_scalars_raw[b * N : (b + 1) * N]

                out = model(s_img, s_c, d_c, s_scal)
                pred_scalars_clamped = torch.clamp(compute_physical_scalars_from_image(out["pred_images"]), min=0.0)

                all_sim_scalars.append(s_scal)
                all_pred_scalars.append(pred_scalars_clamped)
                all_data_scalars.append(d_scal)
                
                # Accumulo per dizionario (usando CPU per risparmiare VRAM)
                for i in range(N):
                    # Definiamo la chiave di binning qui
                    key = (tuple(np.round(s_c[i].cpu().numpy(), 3)), 
                           tuple(np.round(d_c[i].cpu().numpy(), 3)))
                    
                    pair_dict[key].append({
                        'sim_val': s_scal[i].cpu().numpy(),
                        'pred_val': pred_scalars_clamped[i].cpu().numpy(),
                        'data_val': d_scal[i].cpu().numpy()
                    })

    # 2. FILTRAGGIO "BEST-EFFORT" (Nessun filtro rigido, prendiamo i migliori disponibili)
    # Ordiniamo tutte le chiavi trovate per numero di eventi (da quelle con più dati a quelle con meno)
    sorted_keys = sorted(pair_dict.keys(), key=lambda k: len(pair_dict[k]), reverse=True)
    if not sorted_keys:
        print("[CRITICO] Nessuna coppia trovata nel dataset di test.")
        return

    # Selezioniamo le migliori N coppie (senza scartare nulla, prendiamo quello che c'è)
    n_plots = num_sim_ctx_to_sample * num_data_ctx_to_sample
    coppie_selezionate = sorted_keys[:min(n_plots, len(sorted_keys))]
    
    print(f"DEBUG: Trovate {len(sorted_keys)} coppie. Plotting delle {len(coppie_selezionate)} migliori.")
    
    # 3. GENERAZIONE GRIGLIE
    whitelist_recomputed = ["integral_recomputed", "length_recomputed", "width_recomputed"]
    num_scalars = 3 # O calcolato dinamicamente
    
    n_cols = min(3, num_data_ctx_to_sample)
    n_rows = (len(coppie_selezionate) + n_cols - 1) // n_cols

    for idx_var in range(num_scalars):
        var_name = whitelist_recomputed[idx_var]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx_coppia, coppia in enumerate(coppie_selezionate):
            ax = axes_flat[idx_coppia]
            
            # Unpacking della chiave (coppia = (s_ctx, d_ctx))
            s_ctx, d_ctx = np.array(coppia[0]), np.array(coppia[1])
            events = pair_dict[coppia]
            
            s_vals = np.array([e['sim_val'][idx_var] for e in events])
            p_vals = np.array([e['pred_val'][idx_var] for e in events])
            d_vals = np.array([e['data_val'][idx_var] for e in events])

            # Range dinamico
            all_vals = np.concatenate([s_vals, p_vals, d_vals])
            vmin, vmax = np.percentile(all_vals, 1.0), np.percentile(all_vals, 99.0)
            bins = np.linspace(vmin, vmax, 30)

            # Plot Istogrammi
            ax.hist(s_vals, bins=bins, alpha=0.5, histtype="step", linewidth=2, label="SIM", color="tab:blue", density=True)
            ax.hist(p_vals, bins=bins, alpha=0.7, histtype="step", linewidth=2, label="PRED", color="tab:orange", density=True)

            # Plot Dati con Errore
            counts, bin_edges = np.histogram(d_vals, bins=bins)
            counts_density, _ = np.histogram(d_vals, bins=bins, density=True)
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
                        fmt='o', markersize=4, color="black", capsize=2, label="DATA")

            # --------------------------------------------------------
            # 4. TITOLO AD ALTA VISIBILITÀ (FORMATO LATEX)
            # --------------------------------------------------------
            # Spacchettiamo tutte le variabili.
            # Assicurati che gli indici corrispondano all'ordine con cui le salvi nel tensore delle condizioni
            
            sim_str = (r"$\mathbf{SIM:}$ " + 
                       f"z={s_ctx[0]:.1f}, " + 
                       r"$\alpha$=" + f"{s_ctx[1]:.3f}, " + 
                       r"$\lambda$=" + f"{s_ctx[2]:.2f}")
            
            data_str = (r"$\mathbf{DATA:}$ " + 
                        f"z={d_ctx[0]:.1f}, " + 
                        r"$P$=" + f"{d_ctx[1]:.2f}, " + 
                        r"$T$=" + f"{d_ctx[2]:.1f}, " + 
                        r"$H$=" + f"{d_ctx[3]:.2f}")

            # Impostiamo il titolo con un padding per non sovrapporsi al grafico
            ax.set_title(f"{sim_str}\n{data_str}", fontsize=8.5, pad=8, loc='left')
            
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(loc="upper right", fontsize=7)

        # Nascondi assi vuoti
        for i in range(idx_coppia + 1, len(axes_flat)):
            axes_flat[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"griglia_ottimizzata_{var_name}.png"), dpi=120)
        plt.close()


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
