import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class UnpairedTransportDataset(torch.utils.data.Dataset):
    def __init__(self, source_data, target_data, standardize=True, dtype=torch.float32):
        
        # this is to filter out empty datasets after the pre-selections
        valid_sim_keys = [
            k for k, df in source_data.items()
            if len(df) > 0
        ]

        valid_data_keys = [
            k for k, df in target_data.items()
            if len(df) > 0
        ]
        
        self.source_keys = valid_sim_keys #list(source_data.keys())
        self.target_keys = valid_data_keys #list(target_data.keys())
        self.dtype=dtype
        
        self.source_data = source_data
        self.target_data = target_data

        self.standardize = standardize
        # pre-calcolo mu/std per chiave
        self.source_scalers = get_scalers(source_data)
        self.target_scalers = get_scalers(target_data)
        
    def __len__(self):
        return max(len(self.source_keys), len(self.target_keys))

    def __getitem__(self, idx):
        # scegli indipendentemente
        sim_key  = self.source_keys[idx % len(self.source_keys)]
        data_key = random.choice(self.target_keys)

        A_sim_df  = self.source_data[sim_key]
        A_data_df = self.target_data[data_key]

        if len(A_sim_df) == 0:
            print("WARNING! EMPTY SIM KEY:", sim_key)

        if len(A_data_df) == 0:
            print("WARNING! EMPTY DATA KEY:", data_key)
    
        return {
            "A_sim":  torch.tensor(A_sim_df.values, dtype=self.dtype),
            "A_data": torch.tensor(A_data_df.values, dtype=self.dtype),
            "sim_key": torch.tensor(sim_key, dtype=self.dtype),
            "data_key": torch.tensor(data_key, dtype=self.dtype),
            "sim_mu": self.source_scalers[sim_key]["mu"],
            "sim_std": self.source_scalers[sim_key]["std"],
            "data_mu": self.target_scalers[data_key]["mu"],
            "data_std": self.target_scalers[data_key]["std"],
        }

def get_scalers(dataframe_dic):
    scalers = {}
    for key, df in dataframe_dic.items():
        tensor = torch.tensor(df.values, dtype=torch.float32)
        mu = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        # attenzione: evitare std=0
        std[std==0] = 1.0
        scalers[key] = {"mu": mu, "std": std}
    return scalers
    
def to_tensor(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)

    if isinstance(x, pd.DataFrame):
        return torch.tensor(x.values, device=device, dtype=dtype)

    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=dtype)

    raise TypeError(f"Unsupported type: {type(x)}")

def build_val_case(
    src_key,
    tgt_key,
    source_data,
    target_data,
    device="cpu",
    dtype=torch.float32
):
    """
    Costruisce un validation case consistente con il training.
    source_data: 1 pandas.DataFrame OR np.ndarray OR torch.Tensor
    target_data: 1 pandas.DataFrame OR np.ndarray OR torch.Tensor
    """

    # --- distribuzioni ---
    A_sim  = to_tensor(source_data, device, dtype)
    A_data = to_tensor(target_data, device, dtype)

    # --- context ---
    src_key_t = torch.tensor(src_key, device=device, dtype=dtype)
    tgt_key_t = torch.tensor(tgt_key, device=device, dtype=dtype)
    tgt_key_t_reduced = tgt_key_t[..., 1:] # remove Z from the target context
    
    context = torch.cat([src_key_t, tgt_key_t_reduced]).unsqueeze(0)  # (1, C)

    return {
        "A_sim": A_sim,
        "A_data": A_data,
        "context": context
    }

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
        A_sim = sample_from_th2(h_src, self.batch_size)
        A_data = sample_from_th2(h_tgt, self.batch_size)

        # 4) contesto completo (alpha,beta,x,y)
        context = torch.tensor(
            [alpha, beta, x, y],
            dtype=torch.float32
        ).repeat(self.batch_size, 1)

        return {
            "A_sim": A_sim,       # [B,2]
            "A_data": A_data,       # [B,2]
            "context": context    # [B,4]
        }

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
    
