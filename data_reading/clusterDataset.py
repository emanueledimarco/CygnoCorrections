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
        transform=None,
        is_test=False
    ):

        self.n_clusters = n_clusters
        self.transform = transform
        self.is_test = is_test

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

        # ---------------------------------
        # COSTRUIAMO UNA LISTA DETERMINISTICA DI COPPIE (SIM, DATA)
        # ---------------------------------
        self.all_combinations = []
        for z in self.shared_z:
            for s_k in self.sim_keys_by_z[z]:
                for d_k in self.data_keys_by_z[z]:
                    self.all_combinations.append({
                        "z": z,
                        "sim_key": s_k,
                        "data_key": d_k
                    })
        
        # pseudo-length
        self.dataset_length = 100000

    def __len__(self):
        # Se siamo in test, la lunghezza corrisponde esattamente al numero di combinazioni reali!
        if self.is_test:
            return len(self.all_combinations)
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
        if self.is_test:
            # --- MODALITÀ TEST: Determinismo assoluto delle chiavi ---
            combo = self.all_combinations[idx]
            z = combo["z"]
            sim_key = combo["sim_key"]
            data_key = combo["data_key"]
            
            # Fissiamo un seed locale basato sull'indice per rendere riproducibile l'estrazione dei 32 cluster
            rng = np.random.RandomState(idx)
            
            sim_cluster_list = self.sim_dict[sim_key]
            sim_replace = len(sim_cluster_list) < self.n_clusters
            sim_idx = rng.choice(len(sim_cluster_list), self.n_clusters, replace=sim_replace)
            sim_clusters = [sim_cluster_list[i] for i in sim_idx]
            
            data_cluster_list = self.data_dict[data_key]
            data_replace = len(data_cluster_list) < self.n_clusters
            data_idx = rng.choice(len(data_cluster_list), self.n_clusters, replace=data_replace)
            data_clusters = [data_cluster_list[i] for i in data_idx]
            
        else:
            # --- MODALITÀ TRAINING: Stocastico con rng controllato ---
            rng = np.random.RandomState(idx)
            z = rng.choice(self.shared_z)
            
            # Recuperiamo le liste di chiavi disponibili per questo specifico z
            sim_options = self.sim_keys_by_z[z]
            data_options = self.data_keys_by_z[z]
            
            # Scegliamo un INDICE intero casuale per SIM e DATA
            idx_sim = rng.randint(0, len(sim_options))
            idx_data = rng.randint(0, len(data_options))
            
            # Estraiamo la chiave corrispondente (la tupla originale rimane intatta)
            sim_key = sim_options[idx_sim]
            data_key = data_options[idx_data]

            sim_clusters = self.sample_clusters(self.sim_dict[sim_key])
            data_clusters = self.sample_clusters(self.data_dict[data_key])

        return {
            "z": z,
            "sim_cond": sim_key,
            "data_cond": data_key,
            "sim_clusters": sim_clusters,
            "data_clusters": data_clusters,
        }
