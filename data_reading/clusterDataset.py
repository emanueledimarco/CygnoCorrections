import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class ConditionalClusterDataset(Dataset):

    def __init__(
        self,
        pkl_file,
        n_clusters=32,
        min_clusters_per_condition=10,
        transform=None
    ):

        self.n_clusters = n_clusters
        self.transform = transform

        # ---------------------------------
        # load cache
        # ---------------------------------
        with open(pkl_file, "rb") as f:
            dataset_bundle = pickle.load(f)

        self.sim_dict = dataset_bundle["sim"]
        self.data_dict = dataset_bundle["data"]
        self.metadata = dataset_bundle.get("metadata", {})
        self.target_scalars = [v.replace("sc_","") for v in self.metadata["features"]["flow_scalar_variables"]]
        
        # ---------------------------------
        # group keys by z
        # ---------------------------------
        self.sim_keys_by_z = {}
        self.data_keys_by_z = {}

        for key in self.sim_dict:

            z, alpha, lambda_ = key
            
            if len(self.sim_dict[key]) < min_clusters_per_condition:
                continue

            self.sim_keys_by_z.setdefault(z, []).append(key)

        for key in self.data_dict:

            z, P, T, H = key

            if len(self.data_dict[key]) < min_clusters_per_condition:
                continue

            self.data_keys_by_z.setdefault(z, []).append(key)

        # ---------------------------------
        # keep only shared z
        # ---------------------------------
        self.shared_z = sorted(
            set(self.sim_keys_by_z.keys())
            & set(self.data_keys_by_z.keys())
        )

        if len(self.shared_z) == 0:
            raise RuntimeError(
                "No shared z values between sim and data."
            )

        # pseudo-length
        self.dataset_length = 100000

    def __len__(self):
        return self.dataset_length


    def cluster_scalars_to_tensor(self, cluster, keys_to_include=None):
        """
        Estrae gli scalari ordinati alfabeticamente.
        Se 'keys_to_include' è una lista/set di stringhe, estrae SOLO quelle chiavi
        per il calcolo della loss, ignorando le variabili di selezione o i metadati.
        """
        excluded = {"pix", "cond", "meta"}
        values = []
     
        # Selezioniamo le chiavi: o quelle esplicite (white-list) o tutte quelle nel dizionario
        available_keys = cluster.__dict__.keys()
        if keys_to_include is not None:
            # Prendiamo solo l'intersezione tra quelle richieste e quelle realmente presenti
            keys_to_process = [k for k in keys_to_include if k in available_keys]
        else:
            keys_to_process = [k for k in available_keys if k not in excluded]
     
        # Ordiniamo alfabeticamente per garantire il determinismo totale
        for k in sorted(keys_to_process):
            v = cluster.__dict__[k]
            if np.isscalar(v):
                values.append(float(v))
     
        return torch.tensor(values, dtype=torch.float32)
    

    def clusters_to_tensors(self, clusters, keys_to_include=None):

        pix = []
        scalars = []

        for c in clusters:

            p = torch.tensor(
                c.pix,
                dtype=torch.float32
            )

            if self.transform is not None:
                p = self.transform(p)

            pix.append(p)

            scalars.append(self.cluster_scalars_to_tensor(c,keys_to_include)
            )

        return pix, torch.stack(scalars)


    def sample_clusters(self, cluster_list):

        replace = len(cluster_list) < self.n_clusters

        idx = np.random.choice(
            len(cluster_list),
            self.n_clusters,
            replace=replace
        )

        return [cluster_list[i] for i in idx]


    def __getitem__(self, idx):

        # ---------------------------------
        # choose shared z
        # ---------------------------------
        z = random.choice(self.shared_z)

        # ---------------------------------
        # choose sim condition
        # ---------------------------------
        sim_key = random.choice(
            self.sim_keys_by_z[z]
        )

        sim_clusters = self.sample_clusters(
            self.sim_dict[sim_key]
        )

        # ---------------------------------
        # choose data condition
        # ---------------------------------
        data_key = random.choice(
            self.data_keys_by_z[z]
        )

        data_clusters = self.sample_clusters(
            self.data_dict[data_key]
        )

        return {

        "z": z,

        "sim_cond": sim_key,
        "data_cond": data_key,

        "sim_clusters": sim_clusters,
        "data_clusters": data_clusters,

        }
    
