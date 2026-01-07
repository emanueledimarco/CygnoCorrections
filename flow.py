import numpy as np
import torch
import torch.nn as nn
import ROOT

from torch.utils.data import DataLoader
from HistFlowDataset import *

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


class ContextEncoder(torch.nn.Module):
    def __init__(self, context_dim=4, emb_dim=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(context_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, emb_dim),
        )

    def forward(self, context):
        # context: [B, 4] = (α, β, x, y)
        return self.net(context)


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


def train_step(flow, context_encoder, opt, batch):
    A_src = batch["A_src"]
    A_tgt = batch["A_tgt"]
    context = batch["context"]

    cond = context_encoder(context)
    A_out = flow(A_src, cond)

    loss_mmd = mmd_loss(A_out, A_tgt)
    loss_id  = ((A_out - A_src)**2).mean()

    loss = loss_mmd + 0.1 * loss_id

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item(), loss_mmd.item(), loss_id.item()

if __name__ == "__main__":

    toy_sim  = ROOT.TFile.Open("toy_simulation.root")
    toy_data = ROOT.TFile.Open("toy_data.root")

    source_hists = {}
    target_hists = {}
    
    # simulation distributions
    Nx=11
    Ny=11
    for ix in range(Nx):
        for iy in range(Ny):
            source_hists[(ix,iy)] = toy_sim.Get(f"h2_{ix}_{iy}")

    # data distributions
    Na=5
    Nb=5
    for ia in range(Na):
        for ib in range(Nb):
            target_hists[(ia,ib)] = toy_data.Get(f"h2_{ia}_{ib}")

    dataset = TH2UnpairedTransportDataset(
        source_hists,
        target_hists,
        batch_size=32
    )
    
    loader = DataLoader(dataset, batch_size=None, shuffle=True)

    context_encoder = ContextEncoder(context_dim=4, emb_dim=16)
    flow = ConditionalFlow(dim=2, cond_dim=16)

    # Optimizer
    opt = torch.optim.Adam(
    list(flow.parameters()) +
    list(context_encoder.parameters()),
    lr=1e-3
    )

    Nsteps = 2000
    
    for step, batch in enumerate(loader):
        loss, mmd, lid = train_step(flow, context_encoder, opt, batch)

        if step % 200 == 0:
            print(f"{step:5d} | loss {loss:.4f} | MMD {mmd:.4f} | ID {lid:.4f}")

        if step == Nsteps:
            break


    # validation
    # Nominal simulation, e.g. "central pair" of alpha,beta
    x0 = 5
    y0 = 5
    A_sim = sample_from_th2(
        source_hists[(x0, y0)],
        n_samples=50000
    )

    # observed environmental parameters (example within the training range)
    a0 = 2
    b0 = 1
    context = torch.tensor(
        [x0, y0, a0, b0],
        dtype=torch.float32
    ).repeat(len(A_sim), 1)

    flow.eval()
    context_encoder.eval()

    with torch.no_grad():
        cond = context_encoder(context)
        A_corr = flow(A_sim, cond)

    plot_dataset(A_corr,"A_corr")
