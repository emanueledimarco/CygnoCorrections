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
        e_sim = self.sim_encoder(sim_cond)
        if sim_cond.shape[-1] == data_cond.shape[-1] and torch.equal(sim_cond, data_cond):
            e_data = self.sim_encoder(data_cond)
        else:
            e_data = self.data_encoder(data_cond)
        return e_sim, e_data

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return F.relu(out + residual)

class FiLMBlob(nn.Module):
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.fc = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        film_params = self.fc(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
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

class SpatialNoiseInjection(nn.Module):
    """Inietta fluttuazioni stocastiche proporzionali all'intensità del segnale.
    Evita di distruggere lo sfondo o spegnere il cluster."""
    def __init__(self, channels, noise_intensity=0.1):
        super().__init__()
        # Intensità della fluttuazione (es. 10% del valore del pixel)
        self.noise_intensity = noise_intensity

    def forward(self, x):
        if not self.training:
            return x
        # Genera rumore gaussiano standard
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        
        # Iniezione moltiplicativa: il rumore scala con l'attivazione locale delle feature
        # x * (1 + intensity * noise)
        return x * (1.0 + self.noise_intensity * noise)

class FiLMMaskedDecoder(nn.Module):
    def __init__(self, latent_dim=128, cond_total_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Blocco 1: da 4x4 a 8x8 tramite PixelShuffle
        # Per raddoppiare la risoluzione spaziale mantenendo 128 canali in uscita,
        # partiamo da 128 * 4 = 512 canali convoluzionali.
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.ps1 = nn.PixelShuffle(upscale_factor=2) # -> 128 canali, 8x8
        self.noise1 = SpatialNoiseInjection(128)
        self.film1 = FiLMBlob(cond_total_dim, 128)
        self.res1 = ResNetBlock(128)
        
        # Blocco 2: da 8x8 a 16x16
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.ps2 = nn.PixelShuffle(upscale_factor=2) # -> 64 canali, 16x16
        self.noise2 = SpatialNoiseInjection(64)
        self.film2 = FiLMBlob(cond_total_dim, 64)
        self.res2 = ResNetBlock(64)
        
        # Blocco 3: da 16x16 a 32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ps3 = nn.PixelShuffle(upscale_factor=2) # -> 32 canali, 32x32
        self.noise3 = SpatialNoiseInjection(32)
        self.film3 = FiLMBlob(cond_total_dim, 32)
        self.res3 = ResNetBlock(32)
        
        # Blocco 4: da 32x32 a 64x64 (canale singolo di output)
        self.conv4 = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.ps4 = nn.PixelShuffle(upscale_factor=2) # -> 1 canale, 64x64

    def forward(self, h, cond):
        x = self.fc(h).view(-1, 256, 4, 4)
        
        x = self.ps1(self.conv1(x))
        x = self.noise1(x)
        x = self.film1(x, cond)
        x = self.res1(x)
        
        x = self.ps2(self.conv2(x))
        x = self.noise2(x)
        x = self.film2(x, cond)
        x = self.res2(x)
        
        x = self.ps3(self.conv3(x))
        x = self.noise3(x)
        x = self.film3(x, cond)
        x = self.res3(x)
        
        return self.ps4(self.conv4(x))

class CygnoTransportModel(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=64, noise_scale=0.05):
        super().__init__()
        self.encoder = ResNetClusterEncoder(latent_dim)
        self.decoder = FiLMMaskedDecoder(latent_dim, cond_total_dim=cond_dim*2)
        self.cond_encoder = ConditionEncoder()
        self.transport = DifferentialTransport(latent_dim, cond_dim=cond_dim)
        self.noise_scale = noise_scale

    def forward(self, sim_img, sim_cond, data_cond, sim_scalars):
        # 1. Encoding spaziale
        h = self.encoder(sim_img)
        
        # 2. INIEZIONE RUMORE POST-ENCODER (Cura del Mode Collapse / Bias Deterministico)
        if self.training:
            noise = torch.randn_like(h) * self.noise_scale
            h = h + noise
        
        # 3. Embedding contestuale condizionato
        e_sim, e_data = self.cond_encoder(sim_cond, data_cond)
        cond_totale = torch.cat([e_sim, e_data], dim=-1)

        # 4. Trasporto differenziale latente 
        delta_h = self.transport(h, cond_totale)
        h_corr = h + delta_h
        
        # 5. Generazione modulata continua
        pred_img = self.decoder(h_corr, cond_totale)

        return {
            "pred_images": pred_img,
            "latent": h,
            "delta_h": delta_h
        }

class DifferentialTransport(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + (2 * cond_dim), 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, h, cond_total):
        x = torch.cat([h, cond_total], dim=-1)
        raw_delta = self.net(x)
        return self.gamma * raw_delta
    
def compute_physical_scalars_from_image(images, eps=1e-4):
    device = images.device
    H, W = images.shape[-2], images.shape[-1]
    imgs = images.view(-1, H, W) 
    L_batch = imgs.shape[0]      
    
    y_indices, x_indices = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) - H // 2,
        torch.arange(W, dtype=torch.float32, device=device) - W // 2,
        indexing="ij"
    )
    
    x_coords = x_indices.unsqueeze(0) 
    y_coords = y_indices.unsqueeze(0) 
    imgs = torch.clamp(imgs, min=0.0)
    
    # --- A. INTEGRALE ---
    integrals = torch.sum(imgs, dim=[1, 2]) 
    integrals_safe = torch.where(integrals > eps, integrals, torch.tensor(eps, device=device))
    
    # --- B. CENTROIDI ---
    x_c = torch.sum(imgs * x_coords, dim=[1, 2]) / integrals_safe 
    y_c = torch.sum(imgs * y_coords, dim=[1, 2]) / integrals_safe 
    
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

    # --- E. 4 NUOVE METRICHE ESTESE ---
    area = torch.sum(imgs > 0.0, dim=[1, 2]).float() + eps
    density = integrals_safe / area
    
    eccentricity_arg = torch.clamp(1.0 - (lambda_min / (lambda_max + eps)), min=0.0)
    eccentricity = torch.sqrt(eccentricity_arg)
    
    max_pixel = imgs.view(L_batch, -1).max(dim=1)[0]
    relative_peak = max_pixel / integrals_safe
    
    v_x = lambda_max - mu_yy
    v_y = mu_xy
    norm = torch.sqrt(v_x**2 + v_y**2 + eps)
    v_x = (v_x / norm).view(L_batch, 1, 1)
    v_y = (v_y / norm).view(L_batch, 1, 1)
    
    u = x_centered * v_x + y_centered * v_y
    mu_3 = torch.sum(imgs * (u ** 3), dim=[1, 2]) / integrals_safe
    sigma_3 = torch.clamp(lambda_max, min=0.0)**1.5 + eps
    skewness = torch.abs(mu_3 / sigma_3)
    
    return torch.stack([integrals, lengths, widths, density, eccentricity, relative_peak, skewness], dim=1)


def compute_centroids(images, eps=1e-4):
    device = images.device
    H, W = images.shape[-2], images.shape[-1]
    imgs = images.view(-1, H, W)
    L_batch = imgs.shape[0]
    
    y_indices, x_indices = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) - H // 2,
        torch.arange(W, dtype=torch.float32, device=device) - W // 2,
        indexing="ij"
    )
    x_coords = x_indices.unsqueeze(0)
    y_coords = y_indices.unsqueeze(0)
    imgs = torch.clamp(imgs, min=0.0)
    
    integrals = torch.sum(imgs, dim=[1, 2])
    integrals_safe = torch.where(integrals > eps, integrals, torch.tensor(eps, device=device))
    
    x_c = torch.sum(imgs * x_coords, dim=[1, 2]) / integrals_safe
    y_c = torch.sum(imgs * y_coords, dim=[1, 2]) / integrals_safe
    return x_c, y_c

def compute_centroid_loss(sim, pred):
    sim_xc, sim_yc = compute_centroids(sim)
    pred_xc, pred_yc = compute_centroids(pred)
    return torch.mean((sim_xc - pred_xc)**2 + (sim_yc - pred_yc)**2)

def compute_compactness_loss(images, xc, yc):
    N, C, H, W = images.shape
    device = images.device
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device).float()/H, 
                                    torch.arange(W, device=device).float()/W, 
                                    indexing='ij')
    xc_exp = xc.view(N, 1, 1, 1)
    yc_exp = yc.view(N, 1, 1, 1)
    dist_sq = (x_grid - xc_exp)**2 + (y_grid - yc_exp)**2
    weighted_dist = images * dist_sq 
    num = torch.sum(weighted_dist, dim=(1, 2, 3))
    den = torch.sum(images, dim=(1, 2, 3)) + 1e-6
    return (num / den).mean()

def compute_mmd_rbf(X, Y):
    B_X = X.shape[0]
    B_Y = Y.shape[0]
    X = X.view(B_X, -1)
    Y = Y.view(B_Y, -1)
    
    XX = torch.sum((X.unsqueeze(1) - X.unsqueeze(0)) ** 2, dim=-1)
    YY = torch.sum((Y.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1)
    XY = torch.sum((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=-1)
    
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    mmd_loss = 0.0
    for alpha in alphas:
        K_XX = torch.exp(- XX / alpha)
        K_YY = torch.exp(- YY / alpha)
        K_XY = torch.exp(- XY / alpha)
        
        K_XX_sum = K_XX.sum() - torch.trace(K_XX)
        K_YY_sum = K_YY.sum() - torch.trace(K_YY)
        
        term_XX = K_XX_sum / (B_X * (B_X - 1) + 1e-8)
        term_YY = K_YY_sum / (B_Y * (B_Y - 1) + 1e-8)
        term_XY = 2.0 * K_XY.sum() / (B_X * B_Y + 1e-8)
        
        mmd_loss += (term_XX + term_YY - term_XY)
    return torch.clamp(mmd_loss, min=0.0)

def extract_profiles_with_diagonals(img_tensor):
    B, C, H, W = img_tensor.shape
    img = img_tensor.view(B, H, W)
    prof_x = img.sum(dim=1) 
    prof_y = img.sum(dim=2) 
    
    diags_1, diags_2 = [], []
    img_flipped = torch.flip(img, dims=[2])
    for offset in range(-H + 1, W):
        diags_1.append(torch.diagonal(img, offset=offset, dim1=1, dim2=2).sum(dim=1))
        diags_2.append(torch.diagonal(img_flipped, offset=offset, dim1=1, dim2=2).sum(dim=1))
        
    return torch.cat([prof_x, prof_y, torch.stack(diags_1, dim=1), torch.stack(diags_2, dim=1)], dim=1)

def compute_radial_profile(img, bins=20):
    h, w = img.shape[-2:]
    center_y, center_x = h // 2, w // 2
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2).to(img.device)
    max_dist = dist.max()
    bin_edges = torch.linspace(0, max_dist, bins + 1)
    
    profile = []
    for i in range(bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i+1])
        profile.append((img * mask).sum() / (mask.sum() + 1e-6))
    return torch.stack(profile)

def compute_radius_of_gyration(img):
    total_mass = img.sum(dim=(-2, -1))
    h, w = img.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    y = y.to(img.device); x = x.to(img.device)
    yc = (y * img).sum(dim=(-2, -1)) / (total_mass + 1e-6)
    xc = (x * img).sum(dim=(-2, -1)) / (total_mass + 1e-6)
    r2 = ((x - xc.view(-1,1,1))**2 + (y - yc.view(-1,1,1))**2) * img
    return torch.sqrt(r2.sum(dim=(-2, -1)) / (total_mass + 1e-6))


# === COMPUTE LOSS (TV E LAPLACIAN RIMOSSI) ===
def compute_cygno_loss(
    pred, pred_scalars, target_scalars, delta_h, loss_weights, 
    pred_identity=None, sim_images=None
):
    data_clamped = torch.clamp(target_scalars, min=0.0) # Fallback / Safety
    pred_clamped = F.elu(pred) + 1.0

    L_identity = 0.0
    if pred_identity is not None and sim_images is not None:
        L_identity = F.mse_loss(F.elu(pred_identity) + 1.0, sim_images)

    pred_n = pred_clamped / (pred_clamped.sum(dim=(-1, -2), keepdim=True) + 1e-8)
    sim_n  = sim_images / (sim_images.sum(dim=(-1, -2), keepdim=True) + 1e-8)
    
    # Profilo 1D e Radiale
    # Passiamo da MSE a L1 per evitare la regressione verso la media (effetto smooth)
    L_mmd_shape_1d = compute_mmd_rbf(extract_profiles_with_diagonals(pred_n), extract_profiles_with_diagonals(sim_n))
    L_mmd_radial = F.smooth_l1_loss(compute_radial_profile(pred_n), compute_radial_profile(sim_n), beta=0.05)
    L_rg = F.l1_loss(compute_radius_of_gyration(pred_n), compute_radius_of_gyration(sim_n))   # <--- Cambiato in L1
    
    # Shape Anchor
    loss_pixel_matrix = F.smooth_l1_loss(pred_n, sim_n, beta=0.01, reduction='none')
    weight_mask = (sim_n / (sim_n.mean(dim=(-1, -2), keepdim=True) + 1e-8)).detach()
    L_pixel_shape_anchor = (loss_pixel_matrix * weight_mask).mean()
    
    # --- VALIDAZIONE ED ALLINEAMENTO COMPLETO DELLE 7 METRICHE FISICHE ---
    pred_physics_feat = torch.stack([
        torch.log10(pred_scalars[:, 0] + 1.0),   # Integral
        pred_scalars[:, 1] / 10.0,               # Length
        pred_scalars[:, 2] / 10.0,               # Width
        pred_scalars[:, 3] / 5.0,                # Density
        pred_scalars[:, 4],                      # Eccentricity (già 0-1)
        pred_scalars[:, 5] * 10.0,               # Relative Peak
        pred_scalars[:, 6]                       # Skewness
    ], dim=1)

    target_physics_feat = torch.stack([
        torch.log10(target_scalars[:, 0] + 1.0),
        target_scalars[:, 1] / 10.0,
        target_scalars[:, 2] / 10.0,
        target_scalars[:, 3] / 5.0,
        target_scalars[:, 4],
        target_scalars[:, 5] * 10.0,
        target_scalars[:, 6]
    ], dim=1)
        
    L_mmd_physics = compute_mmd_rbf(pred_physics_feat, target_physics_feat)
    L_integral = F.mse_loss(pred_physics_feat[:, 0].mean(), target_physics_feat[:, 0].mean())
    L_transport = (delta_h.pow(2)).mean()
    L_centroid = compute_centroid_loss(sim_images, pred)

    # Combinazione pesata senza computazione TV/Laplace inutile
    loss = (
        loss_weights["mmd_shape_1d"] * L_mmd_shape_1d +
        loss_weights["shape_anchor"] * L_pixel_shape_anchor +
        loss_weights["mmd_physics"] * L_mmd_physics +
        loss_weights["integral"] * L_integral +
        loss_weights["transport"] * L_transport +
        loss_weights["centroid"] * L_centroid +
        loss_weights["radial"] * L_mmd_radial
    )

    loss_dict = {
        "total": loss.item(),
        "mmd_shape_1d": loss_weights["mmd_shape_1d"] * L_mmd_shape_1d.item(),
        "shape_anchor": loss_weights["shape_anchor"] * L_pixel_shape_anchor.item(),        
        "mmd_physics": loss_weights["mmd_physics"] * L_mmd_physics.item(),
        "integral": loss_weights["integral"] * L_integral.item(),
        "transport": loss_weights["transport"] * L_transport.item(),
        "centroid": loss_weights["centroid"] * L_centroid.item(),
        "radial": loss_weights["radial"] * L_mmd_radial.item()
    }
    return loss, loss_dict


def train_epoch(model, loader, optimizer, loss_weights, device="mps", max_batches=None):
    model.train()
    epoch_stats = defaultdict(list)

    for ibatch, batch in enumerate(loader):
        if max_batches is not None and ibatch >= max_batches:
            break

        if ibatch % 10 == 0:
            print(f"\t\t  running ibatch {ibatch}...")

        sim_images = batch["sim_images"].to(device)
        data_images = batch["data_images"].to(device)
        sim_cond = batch["sim_cond"].to(device)
        data_cond = batch["data_cond"].to(device)

        B, N, H, W = sim_images.shape
        sim_images_flat = sim_images.view(B * N, 1, H, W)
        data_images_flat = data_images.view(B * N, 1, H, W)

        # Estraiamo l'intero spettro a 7 parametri in modo differenziabile
        sim_scalars_phys = compute_physical_scalars_from_image(sim_images_flat).view(B, N, -1)
        data_scalars_phys = compute_physical_scalars_from_image(data_images_flat).view(B, N, -1)

        optimizer.zero_grad()
        loss_batch_accumulata = 0.0
        info_batch_accumulato = defaultdict(float)

        for b in range(B):
            s_img = sim_images[b].unsqueeze(1)
            d_img = data_images[b].unsqueeze(1)
            s_cond = sim_cond[b].unsqueeze(0).repeat(N, 1)
            d_cond = data_cond[b].unsqueeze(0).repeat(N, 1)
            s_scal = sim_scalars_phys[b]
            d_scal = data_scalars_phys[b]

            out = model(s_img, s_cond, d_cond, s_scal)
            pred_img = out["pred_images"]
            pred_scalars_phys = compute_physical_scalars_from_image(pred_img)

            out_identity = model(s_img, s_cond, s_cond, s_scal)
            
            loss_evento, info_evento = compute_cygno_loss(
                pred=pred_img, pred_scalars=pred_scalars_phys, target_scalars=d_scal,
                delta_h=out["delta_h"], pred_identity=out_identity["pred_images"],
                sim_images=s_img, loss_weights=loss_weights
            )

            loss_batch_accumulata += loss_evento / B
            for k, v in info_evento.items():
                info_batch_accumulato[k] += v / B

        loss_batch_accumulata.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in info_batch_accumulato.items():
            epoch_stats[k].append(v)
            
    return {k: np.mean(v) for k, v in epoch_stats.items()}

def get_loss_weights(epoch, total_epochs):
    # Ricalibrazione numerica basata sui valori reali dell'epoca 0
    weights_schedule = {
        "mmd_shape_1d": [300.0, 300.0],     # Ok (~600)
        "shape_anchor": [1500.0, 800.0],    # ALZATA: da 50.0 a 1500.0 per forzare l'inclinazione XY
        "mmd_physics":  [80.0, 100.0],       # ABBASSATA: da 4000.0 a 80.0 per togliere la dominanza
        "integral":     [2000.0, 1000.0],    # Bilanciata (~1000)
        "transport":    [0.5, 0.5],          # Lascia respirare il flusso latente
        "centroid":     [10.0, 10.0],        # Ok (~100)
        "radial":       [100000.0, 50000.0], # ALZATA TRADIZIONALMENTE: Huber loss sui profili ha bisogno di una scala enorme
    }
    alpha = min(epoch / total_epochs, 1.0)
    return {k: (v[0] + (v[1] - v[0]) * alpha) for k, v in weights_schedule.items()}

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


from torch.utils.data import DataLoader

def build_dataloader(
        inputfile,
        batch_size=32,
        n_clusters=32,
        shuffle=True,
        is_test=False,
        num_workers=0
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
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=make_cygno_collate_fn(dataset)
    )

    return dataset, loader

def train_model(inputfile, outputfile, checkpoint_path=None, epochs=50):
    
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

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"===> Loading checkpoint '{checkpoint_path}'...")
    
        # Carica su CPU o sulla GPU corrente
        state_dict = torch.load(checkpoint_path, map_location=device)
    
        # Ripristina i pesi del modello
        model.load_state_dict(state_dict)

        print(f"===> Resuming successfully from {checkpoint_path}")
    else:
        print("===> Starting training from scratch (or checkpoint not found)")

    
    train_history = {
        "total": [],
        "mmd_shape_1d": [],
        "mmd_physics": [],
        "integral": [],
        "transport": [],
        "centroid": [],
        "radial": [],
    }
    
    print(f"Initialized the model. Now start the training on the device: {device}")
    best_val_loss = float('inf')
    for epoch in range(epochs):

        print(f"\t|Start epoch n. {epoch}...")

        # 1. Ottieni i pesi dinamici per questa epoca
        gammas = get_loss_weights(epoch, epochs)
        print(f"\t\t--> Will use the following loss weights: {gammas}")
        
        stats = train_epoch(
            model,
            loader,
            optimizer,
            gammas,
            device=device,
            max_batches=30
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

        if stats["total"] < best_val_loss:
            best_val_loss = stats["total"]
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        },
                       f"{outputfile.replace('last','best')}")
            print(f"Nuovo miglior modello salvato all'epoca {epoch} con loss = {stats['total']}")
                
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                },
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

        # 1. Carica il file .pt generico
        state = torch.load(
            model_or_path,
            map_location=device
        )

        # Controllo intelligente del formato
        if isinstance(state, dict) and 'model_state_dict' in state:
            model_state = state['model_state_dict']
            current_epoch = state['epoch']
            print(f"===> Rilevato checkpoint completo. Estraggo 'model_state_dict', all'epoca {current_epoch}")
        else:
            print("===> Rilevato state_dict puro (vecchio formato). Carico direttamente...")
            model_state = state

        model.load_state_dict(
            model_state
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
    
    B, N, H, W = sim.shape

    sim_flat = sim.view(B * N, 1, H, W)
    data_flat = data.view(B * N, 1, H, W)

    sim_cond_flat = (sim_cond.repeat_interleave(N, dim=0))
    data_cond_flat = (data_cond.repeat_interleave(N, dim=0))

    sim_scalars_flat = compute_physical_scalars_from_image(sim_flat)

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

    test_cond_sim_labels = []
    test_cond_data_labels = []

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
            sim_scalars_flat = torch.clamp(compute_physical_scalars_from_image(sim_flat), min=0.0).view(B, N, -1)
            
            # Espandiamo le condizioni per il match flat
            sim_cond_flat = sim_cond.repeat_interleave(N, dim=0)
            data_cond_flat = data_cond.repeat_interleave(N, dim=0)

            # Forward
            out = model(sim_flat, sim_cond_flat, data_cond_flat, sim_scalars_flat)
            pred_clamped_flat = F.elu(out["pred_images"]) + 1.0
            threshold_value = 0. 
            pred_clamped_flat = torch.where(pred_clamped_flat > threshold_value, pred_clamped_flat, torch.zeros_like(pred_clamped_flat))
            # 3. Reshape finale a 4D delle immagini post-soglia
            pred_images = pred_clamped_flat.view(B, N, H, W)

            pred_scalars = torch.clamp(compute_physical_scalars_from_image(pred_images), min=0.0).view(B, N, -1)
            
            # Scegliamo il primo sotto-cluster (idx=0) di questo specifico batch
            test_sims.append(sim[0, 0].cpu())
            test_preds.append(pred_images[0, 0].cpu())
            test_datas.append(batch["data_images"][0, 0].cpu())

            # Prendiamo il primo elemento del batch (indice 0)
            sim_label = f" z={sim_cond[0, 0].item():.1f}cm,\n$\\alpha$={sim_cond[0, 1].item():.2f},\n$\\lambda$={sim_cond[0, 1]:.2f}mm)"
            test_cond_sim_labels.append(sim_label)
            data_label = f" z={data_cond[0, 0].item():.1f}cm,\nP={data_cond[0, 1].item():.3f}bar,\nT={data_cond[0, 2].item():.1f},\nH={data_cond[0, 3].item():.1f}?"
            test_cond_data_labels.append(data_label)
            
    # Ora disegnamo le 10 colonne, ognuna corrispondente a un BATCH differente
    fig, ax = plt.subplots(nrows=3, ncols=10, figsize=(20, 9))
    
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
        
        ax[0,i].imshow(test_sims[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap)
        ax[0,i].set_title(f"SIM (Batch {i})")
        label = test_cond_sim_labels[i]
        ax[0,i].text(0.5, -0.3, label, 
                     transform=ax[0, i].transAxes, 
                     ha='center', va='top', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))
        
        ax[1,i].imshow(test_preds[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap)
        ax[1,i].set_title(f"CORR (Batch {i})")
        label = test_cond_data_labels[i]
        ax[1,i].text(0.5, -0.3, label, 
                     transform=ax[1, i].transAxes, 
                     ha='center', va='top', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))
        
        # Nota: se i DATI reali hanno un guadagno intrinseco totalmente diverso, 
        # conviene lasciargli il suo vmax per studiare la shape, altrimenti mettiamo vmax anche qui
        im_data = ax[2,i].imshow(test_datas[i], origin="lower", vmin=0, vmax=vmax, cmap=cmap)
        ax[2,i].set_title(f"DATA (Batch {i})")
     
        for row in range(3):
            cbar = fig.colorbar(im_data, ax=ax[row,i], fraction=0.046, pad=0.05)
            cbar.ax.tick_params(labelsize=8)
            if i == 0:
                cbar.set_label("Counts", fontsize=8)
            
    plt.tight_layout()
    cluster_test_path = os.path.join(output_dir, "clusters10_test.png")
    plt.savefig(cluster_test_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\t--> Griglia con 10 clusters salvata con successo in: {cluster_test_path}")

    #run_sampled_and_detailed_test(model,loader,device,max_batches=100,output_dir="plot/validation_plots",save_individual_plots=False)

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
