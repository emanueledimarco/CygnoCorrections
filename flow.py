import numpy as np
import torch
import torch.nn as nn
import ROOT

from torch.utils.data import DataLoader
from HistFlowDataset import *

def pair_grid(x,y,Ny):
    return x * Ny + y


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim, hidden=64, mask=None):
        super().__init__()

        assert mask is not None, "mask must be provided"
        self.register_buffer("mask", mask)

        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim * 2)
        )

    def forward(self, x, cond):
        """
        x:    [B, dim]        input A
        cond: [B, cond_dim]  embedding(x,y)
        """

        # 1) parte che NON viene trasformata
        x_masked = x * self.mask

        # 2) input alla rete che predice scale e shift
        h = torch.cat([x_masked, cond], dim=1)

        # 3) ottieni scale (s) e shift (t)
        s, t = self.net(h).chunk(2, dim=1)

        # 4) applica maschera: solo le dimensioni non mascherate
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)

        # 5) trasformazione affine
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)

        return y
    
class ConditionalFlow(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()

        masks = [
            torch.tensor([1., 0.]),
            torch.tensor([0., 1.]),
            torch.tensor([1., 0.])
        ]

        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(
                dim=dim,
                cond_dim=cond_dim,
                hidden=64,
                mask=m
            )
            for m in masks
        ])

    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        return x


class XYEmbedding(nn.Module):
    def __init__(self, n_xy, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(n_xy, emb_dim)

    def forward(self, xy_id):
        return self.emb(xy_id)

def mmd_loss(x, y, sigma=1.0):
    """
    x: [B, d]  (output del flow)
    y: [B, d]  (target)
    sigma:     kernel bandwidth
    """

    # distanza quadratica tra tutti i punti
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2

    # kernel gaussiano
    k_xx = torch.exp(-xx / (2 * sigma ** 2))
    k_yy = torch.exp(-yy / (2 * sigma ** 2))
    k_xy = torch.exp(-xy / (2 * sigma ** 2))

    # MMD^2
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

def train_step(
    flow,
    embedder,
    optimizer,
    A_src,
    A_tgt,
    xy_id,
    lambda_id=0.1,
    sigma_mmd=1.0
):
    """
    flow:      ConditionalFlow
    embedder:  XYEmbedding
    optimizer: torch.optim.Optimizer
    A_src:     [B,2] float tensor
    A_tgt:     [B,2] float tensor
    xy_id:     [B]   long tensor
    """

    # 1) ottieni embedding (x,y)
    cond = embedder(xy_id)

    # 2) applica il flow
    A_out = flow(A_src, cond)

    # 3) MMD loss (matching distribuzioni)
    loss_mmd = mmd_loss(A_out, A_tgt, sigma=sigma_mmd)

    # 4) Identity regularization
    loss_id = ((A_out - A_src) ** 2).mean()

    # 5) Loss totale
    loss = loss_mmd + lambda_id * loss_id

    # 6) Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_mmd.item(), loss_id.item()

if __name__ == "__main__":

    toy_sim  = ROOT.TFile.Open("toy_simulation.root")
    toy_data = ROOT.TFile.Open("toy_data.root")

    source_hists = {}
    xy_to_id = {}
    
    Nx=11
    Ny=11
    for ix in range(Nx):
        for iy in range(Ny):
            source_hists[(ix,iy)] = toy_sim.Get(f"h2_{ix}_{iy}")
            xy_to_id[(ix,iy)] = pair_grid(ix,iy,Ny)

    target_hist = toy_data.Get(f"h2_0_0")

    #test_sampling(source_hists[(0,0)])
    
    dataset = TH2FlowDataset(
        source_hists,
        target_hist,
        xy_to_id,
        batch_size=32
    )
    
    loader = DataLoader(dataset, batch_size=None)


    # creare il modello
    n_xy = len(xy_to_id) # number of pairs of sim params

    # Embedder
    emb_dim = 12 # to be optimized
    embedder = XYEmbedding(
        n_xy=n_xy,
        emb_dim=emb_dim
    )

    # Flow
    flow = ConditionalFlow(
        dim=2,
        cond_dim=emb_dim
    )

    # Device (testd on CPU only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow.to(device)
    embedder.to(device)

    # Optimizer (both flow and embedder parameters)
    opt = torch.optim.Adam(
        list(flow.parameters()) + list(embedder.parameters()),
        lr=1e-3
    )

    n_steps = 5000

    for step, (A_src, A_tgt, xy_id) in enumerate(loader):

        if step >= n_steps:
            break

        # manda tutto su device
        A_src = A_src.to(device)
        A_tgt = A_tgt.to(device)
        xy_id = xy_id.to(device)

        loss, loss_mmd, loss_id = train_step(
            flow,
            embedder,
            opt,
            A_src,
            A_tgt,
            xy_id,
            lambda_id=0.1
        )

        if step % 200 == 0:
            print(
                f"step {step:5d} | "
                f"loss {loss:.4f} | "
                f"MMD {loss_mmd:.4f} | "
                f"ID {loss_id:.4f}"
            )

    # validate the training
    with torch.no_grad():
        A_new = flow(A_src, embedder(xy_id))

    h_out = ROOT.TH2D("out", "out",
                      100, -3, 5,
                      100, -5, 5)
    
    A = A_new.cpu().numpy()
    for x, y in A:
        h_out.Fill(x, y)

    c = ROOT.TCanvas("c","",600,600)
    h_out.Draw("colz")
    c.SaveAs("A_out.pdf")
    
