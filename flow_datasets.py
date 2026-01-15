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
