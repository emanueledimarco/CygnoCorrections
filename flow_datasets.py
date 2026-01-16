import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class UnpairedTransportDataset(Dataset):
    def __init__(self, source_data, target_data, device="cpu", dtype=torch.float32):
        """
        source_data: dict[key -> pandas.DataFrame OR np.ndarray OR torch.Tensor]
        target_data: dict[key -> pandas.DataFrame OR np.ndarray OR torch.Tensor]
        """
        self.device = device
        self.dtype = dtype

        self.src_keys = list(source_data.keys())
        self.tgt_keys = list(target_data.keys())

        self.source = {k: self._to_tensor(v) for k, v in source_data.items()}
        self.target = {k: self._to_tensor(v) for k, v in target_data.items()}

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(self.device, self.dtype)
        return torch.tensor(x.values, dtype=self.dtype, device=self.device)

    def __len__(self):
        # lunghezza = max, oppure numero di accoppiamenti che vuoi
        return max(len(self.src_keys), len(self.tgt_keys))

    def __getitem__(self, idx):
        # sampling indipendente (unpaired!)
        src_key = self.src_keys[idx % len(self.src_keys)]
        tgt_key = self.tgt_keys[torch.randint(len(self.tgt_keys), (1,)).item()]

        return {
            "A_sim": self.source[src_key],
            "A_data": self.target[tgt_key],
            "sim_key": torch.tensor(src_key, device=self.device),
            "data_key": torch.tensor(tgt_key, device=self.device),
        }
    
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

    context = torch.cat([src_key_t, tgt_key_t]).unsqueeze(0)  # (1, C)

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
    
