import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from data_reading.clusterDataset import ConditionalClusterDataset
from data_reading.read_data_2D import make_cygno_collate_fn

class ConditionEncoder(nn.Module):

    def __init__(
        self,
        emb_dim=64
    ):

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

    def forward(
        self,
        sim_cond,
        data_cond
    ):

        e_sim = self.sim_encoder(
            sim_cond
        )

        e_data = self.data_encoder(
            data_cond
        )

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

    def forward(
        self,
        x
    ):

        h = self.net(x)

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

        self.net = nn.Sequential(

            nn.Linear(latent_dim + cond_dim, 256),
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

    def forward(self, h, delta_e):

        x = torch.cat([h, delta_e], dim=-1)

        raw_delta = self.net(x)

        # ---------------------------------------
        # normalize update scale
        # ---------------------------------------
        raw_delta = raw_delta / (raw_delta.std(dim=-1, keepdim=True) + 1e-6)

        # ---------------------------------------
        # controlled residual update
        # ---------------------------------------
        delta_h = self.gamma * raw_delta

        return delta_h
    
class CygnoTransportModel(
    nn.Module
):

    def __init__(
        self,
        latent_dim=128
    ):

        super().__init__()

        self.encoder = (
            ClusterEncoder(
                latent_dim
            )
        )

        self.cond_encoder = (
            ConditionEncoder()
        )

        self.transport = (
            DifferentialTransport(
                latent_dim
            )
        )

        self.decoder = (
            ClusterDecoder(
                latent_dim
            )
        )

    def forward(
        self,
        sim_img,
        sim_cond,
        data_cond
    ):

        # -------------------
        # encode image
        # -------------------
        h = self.encoder(
            sim_img
        )

        # -------------------
        # encode conditions
        # -------------------
        e_sim, e_data = (
            self.cond_encoder(
                sim_cond,
                data_cond
            )
        )

        delta_e = (
            e_data - e_sim
        )

        # -------------------
        # residual transport
        # -------------------
        delta_h = (
            self.transport(
                h,
                delta_e
            )
        )

        h_corr = h + delta_h

        # -------------------
        # decode
        # -------------------
        pred = self.decoder(
            h_corr
        )

        return {

            "pred": pred,

            "latent": h,

            "delta_h": delta_h,

            "corrected_latent":
                h_corr
        }


def forward_test(inputfile):

    dataset = ConditionalClusterDataset(
        pkl_file=inputfile,
        n_clusters=32
    )
    
    loader = DataLoader(
        dataset,
        batch_size=2,
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

def image_to_scalars(img):

    """
    img: [B,N,H,W]
    """

    integral = img.sum(dim=(-1, -2))

    mean = img.mean(dim=(-1, -2))

    rms = img.std(dim=(-1, -2))

    nhits = (img > 0).float().sum(
        dim=(-1, -2)
    )

    return torch.stack([
        integral,
        mean,
        rms,
        nhits
    ], dim=-1)


def spatial_gradient_loss(pred, target):

    # pred: [B, N, H, W]
    # target: [B, N, H, W]

    B, N, H, W = pred.shape

    pred = pred.view(B*N, 1, H, W)
    target = target.view(B*N, 1, H, W)

    dx_pred = pred[..., :, 1:] - pred[..., :, :-1]
    dy_pred = pred[..., 1:, :] - pred[..., :-1, :]

    dx_tgt = target[..., :, 1:] - target[..., :, :-1]
    dy_tgt = target[..., 1:, :] - target[..., :-1, :]

    loss = (
        (dx_pred - dx_tgt).pow(2).mean()
        +
        (dy_pred - dy_tgt).pow(2).mean()
    )

    return loss

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


# === COMPLETE LOSS FUNCTION ===
# A) distribution matching
# B) physics loss
# C) scalar auxiliary loss
# D) latent regularization
def compute_cygno_loss(
    pred,
    data,
    data_scalars,
    delta_h
):

    # clamp of the intensity
    pred = F.softplus(pred)
    
    pred_n = pred / (pred.sum(dim=(-1,-2), keepdim=True) + 1e-8)
    data_n = data / (data.sum(dim=(-1,-2), keepdim=True) + 1e-8)
    
    # --------------------------------
    # MMD-like feature matching
    # --------------------------------
    def features(x):

        return torch.stack([

            x.sum(dim=(-1, -2)),
            x.mean(dim=(-1, -2)),
            x.std(dim=(-1, -2))

        ], dim=-1)

    pred_n_feat = features(pred_n)
    data_n_feat = features(data_n)

    L_mmd = (
        pred_n_feat.mean(0)
        -
        data_n_feat.mean(0)
    ).pow(2).mean()

    # normalize to the number of elements
    L_mmd = L_mmd / pred_n.shape[0]
    
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
    pred_scalars = image_to_scalars(
        pred
    )

    target_scalars = data_scalars[
        ...,
        :4
    ]

    L_aux = F.mse_loss(
        pred_scalars,
        target_scalars
    )

    # --------------------------------
    # latent near-identity
    # --------------------------------
    L_transport = (
        delta_h.pow(2)
    ).mean()

    
    # --------------------------------
    # spatial loss
    # --------------------------------
    L_spatial = spatial_gradient_loss(
        pred_n,
        data_n
    )

    # ------------------------------------------------------------
    # total variation loss and smoothness (to reduce pixels jumps)
    # ------------------------------------------------------------
    L_tv = total_variation(pred)
    L_lap = laplacian_smoothness(pred)
    
    # --------------------------------
    # final weighted loss
    # --------------------------------
    loss = (

        0.5 * L_mmd
        +
        0.1 * L_integral
        +
        0.1 * L_rms
        +
        0.001 * L_aux
        +
        0.01 * L_transport
        +
        1.0 * L_spatial
        +
        1.0 * L_tv
        +
        0.5 * L_lap
    )

    loss_dict = {
        "total": loss.item(),
        "mmd": L_mmd.item(),
        "integral": L_integral.item(),
        "rms": L_rms.item(),
        "aux": L_aux.item(),
        "transport": L_transport.item(),
        "spatial": L_spatial.item(),
        "totvar": L_tv.item(),
        "laplace": L_tv.item()        
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
        "spatial": [],
        "totvar": [],
        "laplace": [],
        "delta_h": []
    }
    

    print(f"\n\tNumber of batches in this epoch: {len(loader)}")

    for ibatch, batch in enumerate(loader):
        
        if (max_batches is not None
            and ibatch >= max_batches
            ):
            break

        if ibatch % 10 == 0:
            print (f"\t\t  running ibatch {ibatch}...")
    
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

        data_scalars = batch[
            "data_scalars"
        ].to(device)

        B, N, H, W = sim.shape

        sim = sim.view(
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

        out = model(
            sim,
            sim_cond,
            data_cond
        )

        pred = out[
            "pred"
        ].view(
            B,
            N,
            H,
            W
        )

        #print(f"\t\t\t ---> pred sum: {pred.sum().item()}, data sum: {data.sum().item()}")
        
        loss, info = (
            compute_cygno_loss(
                pred,
                data,
                data_scalars,
                out["delta_h"]
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
            "spatial"
        ].append(
            info["spatial"]
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
def train_model(inputfile,outputfile,epochs=20):
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    _, loader = build_dataloader(
        inputfile
    )

    model = (
        CygnoTransportModel()
        .to(device)
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
        "spatial": [],
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
            max_batches=50
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
    _, loader = (
        build_dataloader(
            inputfile,
            batch_size=1
        )
    )

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

    pred = out[
        "pred"
    ].view(
        B,
        N,
        H,
        W
    )

    # -----------------------
    # visual test
    # -----------------------
    import matplotlib.pyplot as plt
    
    idx = np.random.randint(N)

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(15,5)
    )

    ax[0].imshow(
        sim[0,idx].cpu(),
        origin="lower"
    )
    ax[0].set_title("SIM")

    ax[1].imshow(
        pred[0,idx].cpu(),
        origin="lower"
    )
    ax[1].set_title("CORRECTED")

    ax[2].imshow(
        data[0,idx].cpu(),
        origin="lower"
    )
    ax[2].set_title("DATA")

    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(
        10,
        3,
        figsize=(9,20)
    )
    
    for i in range(10):
        
        idx = np.random.randint(N)
        
        vmax = max(
            sim[0,idx].max().item(),
            pred[0,idx].max().item(),
            data[0,idx].max().item()
        )

        ax[i,0].imshow(
            sim[0,idx].cpu(),
            origin="lower",
            # vmin=0,
            # vmax=vmax
        )

        ax[i,1].imshow(
            pred[0,idx].cpu(),
            origin="lower",
            #vmin=0,
            #vmax=vmax
        )

        ax[i,2].imshow(
            data[0,idx].cpu(),
            origin="lower",
            #vmin=0,
            #vmax=vmax
        )

    plt.tight_layout()
    fig.savefig(
        "debug_cluster_corr.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.show()

