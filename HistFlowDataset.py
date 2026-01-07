import numpy as np
import torch
import random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def sample_from_th2(th2, n_samples):
    nx = th2.GetNbinsX()
    ny = th2.GetNbinsY()

    # bin contents
    weights = np.array([
        th2.GetBinContent(ix+1, iy+1)
        for ix in range(nx)
        for iy in range(ny)
    ])

    weights = np.clip(weights, 0, None)
    prob = weights / weights.sum()

    # sample flat bin index
    bin_indices = np.random.choice(len(prob), size=n_samples, p=prob)

    xs, ys = [], []
    for idx in bin_indices:
        ix = int(idx // ny)
        iy = int(idx % ny)

        xlow = th2.GetXaxis().GetBinLowEdge(ix+1)
        xup  = th2.GetXaxis().GetBinUpEdge(ix+1)
        ylow = th2.GetYaxis().GetBinLowEdge(iy+1)
        yup  = th2.GetYaxis().GetBinUpEdge(iy+1)

        xs.append(np.random.uniform(xlow, xup))
        ys.append(np.random.uniform(ylow, yup))

    return torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.float32)

def test_sampling(hist,nevents=10000):
    A = sample_from_th2(hist, nevents)
    plt.scatter(A[:,0], A[:,1], s=1)
    plt.show()

class TH2FlowDataset(Dataset):
    def __init__(self, source_hists, target_hist, xy_to_id,
                 batch_size=32):
        self.source_hists = source_hists
        self.target_hist = target_hist
        self.xy_to_id = xy_to_id
        self.batch_size = batch_size

        self.xy_keys = list(source_hists.keys())

    def __len__(self):
        # lunghezza fittizia: numero di batch per epoca
        return 1000

    def __getitem__(self, idx):
        # 1) scegli una coppia (x,y) a caso
        xy = self.xy_keys[np.random.randint(len(self.xy_keys))]
        xy_id = self.xy_to_id[xy]

        # 2) campiona B eventi source e target
        A_src = sample_from_th2(
            self.source_hists[xy],
            self.batch_size
        )

        A_tgt = sample_from_th2(
            self.target_hist,
            self.batch_size
        )

        xy_id = torch.full(
            (self.batch_size,),
            xy_id,
            dtype=torch.long
        )

        return A_src, A_tgt, xy_id

    def get_full_source(self, xy):
        hist = self.source_hists[xy]
        N = int(hist.GetEntries())
        return sample_from_th2(hist, N)

# The following if there are two distributions:
# source_hists[(alpha,beta)] (lambda_ass,alpha)
# target_hists[(x,y)] (P,T)
# unknown relationship among (alpha,beta) <=> (x,y)

class TH2UnpairedTransportDataset(Dataset):
    def __init__(
        self,
        source_hists,   # dict[(alpha,beta)] -> TH2D
        target_hists,   # dict[(x,y)] -> TH2D
        batch_size=32
    ):
        self.source_keys = list(source_hists.keys())
        self.target_keys = list(target_hists.keys())

        self.source_hists = source_hists
        self.target_hists = target_hists

        self.batch_size = batch_size

    def __len__(self):
        # lunghezza fittizia
        return 20000

    def __getitem__(self, idx):
        # 1) scegli UNA simulazione a caso
        alpha, beta = random.choice(self.source_keys)
        h_src = self.source_hists[(alpha, beta)]

        # 2) scegli UN contesto dati a caso
        x, y = random.choice(self.target_keys)
        h_tgt = self.target_hists[(x, y)]

        # 3) campiona eventi
        A_src = sample_from_th2(h_src, self.batch_size)
        A_tgt = sample_from_th2(h_tgt, self.batch_size)

        # 4) contesto completo (alpha,beta,x,y)
        context = torch.tensor(
            [alpha, beta, x, y],
            dtype=torch.float32
        ).repeat(self.batch_size, 1)

        return {
            "A_src": A_src,       # [B,2]
            "A_tgt": A_tgt,       # [B,2]
            "context": context    # [B,4]
        }
