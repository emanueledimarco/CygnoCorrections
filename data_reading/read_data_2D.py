# Loads and treat the reconstruction tree data and simulation
# test basic plotting with python -m data_reading.read_data 

import os
import pickle
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

import yaml
from yaml import Loader
import json
import hashlib

import torch
from torch.utils.data import DataLoader

from data_reading.cluster import *
from data_reading.clusterDataset import ConditionalClusterDataset

selection_cfg = {
    "integral_min": 2000,
    "integral_max": 50000,
    "x_min": 500,
    "x_max": 2000,
    "y_min": 500,
    "y_max": 2000,
    "min_npix": 500,
    "n_hits": 200
}

def read_data_and_save(conf):

    #loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    stream = open("flow_configuration.yaml", 'r')
    dictionary = yaml.load(stream,Loader)

    with open("cluster_training_list.json", "r") as file:
        json_data = json.load(file)
        
    variables = json_data["var_scalar_list"]
    print("List of the veriables used in the flow for selection/validation:   ", variables)
    
    print(f"Now filling the datasets for the simulation and data. It applies the selection and converts them to panda DFs.  Since many files are involved, it takes time...")

    cachedir = "data/cache"
    sim_clusters_dict = defaultdict(list)
    data_clusters_dict = defaultdict(list)
    
    sim_map  = dictionary["data_inputs"]["sim_map"]
    data_map = dictionary["data_inputs"]["data_map"]
    
    maps = dict(zip(["sim","data"],[sim_map,data_map]))
    for k,m in maps.items():
        with open(m) as f:
            raw_map_dic = yaml.safe_load(f)
            map_dic = {tuple(map(float, k.split(","))): v for k, v in raw_map_dic.items()}
            #print(map_dic)
            if k=="sim":
                print("\t==> Simulation now...")
                for mapkey,files in map_dic.items():
                    sim_clusters_dict[mapkey] = build_dataset_from_files(files, variables, mapkey, isdata=False)
                    #break
            else:
                print("\t==> Data now...")
                for mapkey,files in map_dic.items():
                    data_clusters_dict[mapkey] = build_dataset_from_files(files, variables, mapkey, isdata=True, selection_cfg=selection_cfg)
                    #break

    # save some metadata information
    metadata = {
        "version": conf,
        "description": "CYGNO cluster dataset for sim-data shape translation",
        
    }
    metadata["features"] = {
        "pix": ["x_centered", "y_centered", "charge"],
        "scalars": variables
    }

    metadata["conditioning"] = {
        "sim": ["z", "alpha", "lambda"],
        "data": ["z", "P", "T", "H"],
        "shared_latent": ["z"]
    }

    metadata["stats"] = {
        "n_clusters_sim": sum(len(v) for v in sim_clusters_dict.values()),
        "n_clusters_data": sum(len(v) for v in data_clusters_dict.values()),
    }

    metadata["keys"] = {
        "sim_keys": list(sim_clusters_dict.keys()),
        "data_keys": list(data_clusters_dict.keys()),
    }

    metadata["dataset_hash"] = hashlib.md5(str(metadata).encode()).hexdigest()

    dataset_bundle = {
        "data": data_clusters_dict,
        "sim": sim_clusters_dict,
        "metadata": metadata
    }
    
    os.makedirs(cachedir, exist_ok=True)
    with open(f"{cachedir}/cygno_clusters_dataset.pkl", "wb") as f:
        pickle.dump(dataset_bundle, f, protocol=4)
    

def inspect_cluster_set(clusters, name=""):

    print(f"\n{name} clusters: {len(clusters)}")

    c0 = clusters[0]

    print("Cluster fields:")

    for k, v in c0.__dict__.items():

        if isinstance(v, np.ndarray):
            print(
                f"  {k}: ndarray "
                f"shape={v.shape} "
                f"dtype={v.dtype}"
            )

        else:
            print(
                f"  {k}: {type(v).__name__} "
                f"value={v}"
            )
            

def make_cygno_collate_fn(dataset):
    
    def cygno_collate_fn(batch):

        out = {}
     
        # -----------------------------
        # stack simple tensors
        # -----------------------------
        out["z"] = torch.tensor(
            [b["z"] for b in batch],
            dtype=torch.float32
        )
     
        out["sim_cond"] = [
            torch.tensor(
                b["sim_cond"],
                dtype=torch.float32
            )
            for b in batch
        ]
     
        out["data_cond"] = [
            torch.tensor(
                b["data_cond"],
                dtype=torch.float32
            )
            for b in batch
        ]
     
        out["sim_cond"] = torch.nn.utils.rnn.pad_sequence(
            out["sim_cond"],
            batch_first=True
        )
     
        out["data_cond"] = torch.nn.utils.rnn.pad_sequence(
            out["data_cond"],
            batch_first=True
        )
     
        # -----------------------------
        # keep raw clusters
        # -----------------------------
        out["sim_clusters"] = [
            b["sim_clusters"]
            for b in batch
        ]
     
        out["data_clusters"] = [
            b["data_clusters"]
            for b in batch
        ]
     
        # -----------------------------
        # images
        # -----------------------------
        sim_images = []
        data_images = []
     
        sim_scalars = []
        data_scalars = []
     
        for b in batch:
     
            sim_imgs = []
            sim_sca = []
     
            for c in b["sim_clusters"]:
     
                sim_imgs.append(
                    torch.tensor(
                        c.to_image(64),
                        dtype=torch.float32
                    )
                )
     
                sim_sca.append(
                    dataset.cluster_scalars_to_tensor(c)
                )
     
            sim_images.append(
                torch.stack(sim_imgs)
            )
     
            sim_scalars.append(
                torch.stack(sim_sca)
            )
     
            data_imgs = []
            data_sca = []
     
            for c in b["data_clusters"]:
     
                data_imgs.append(
                    torch.tensor(
                        c.to_image(64),
                        dtype=torch.float32
                    )
                )
     
                data_sca.append(
                    dataset.cluster_scalars_to_tensor(c)
                )
     
            data_images.append(
                torch.stack(data_imgs)
            )
     
            data_scalars.append(
                torch.stack(data_sca)
            )
     
        out["sim_images"] = torch.stack(
            sim_images
        )
     
        out["data_images"] = torch.stack(
            data_images
        )
     
        out["sim_scalars"] = torch.stack(
            sim_scalars
        )
     
        out["data_scalars"] = torch.stack(
            data_scalars
        )
     
        return out

    return cygno_collate_fn
    
def data_loader_test(dataset):
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=make_cygno_collate_fn(dataset)
    )

    batch = next(iter(loader))

    print("\n===== BATCH TEST =====")
    print("z:", batch["z"])
    print("sim_cond:", batch["sim_cond"].shape)
    print("data_cond:", batch["data_cond"].shape)


def image_test(sample):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(
        sample["sim_clusters"][0].to_image(64),
        origin="lower"
    )
    ax[0].set_title("SIM")
    
    ax[1].imshow(
        sample["data_clusters"][0].to_image(64),
        origin="lower"
    )
    ax[1].set_title("DATA")
    
    plt.show()
    
def integrity_tests(inputfile):
        print(f"Reading source_data and target_data from pre-selected cluster datasets in {inputfile}")

        dataset = ConditionalClusterDataset(
            pkl_file=inputfile,
            n_clusters=32
        )

        print("\n===== DATASET INFO =====")
        print("Shared z values:", dataset.shared_z)
        print("Number of sim keys:", sum(len(v) for v in dataset.sim_keys_by_z.values()))
        print("Number of data keys:", sum(len(v) for v in dataset.data_keys_by_z.values()))
        print("Dataset length (virtual):", len(dataset))

        sample = dataset[0]

        print("\n\t ***** SINGLE SAMPLE TEST *****")
        print("\n===== SAMPLE KEYS =====")
        print(sample.keys())
        
        print("\n===== BASIC SHAPES =====")

        print("z:", sample["z"])
        print("sim_cond:", sample["sim_cond"])
        print("data_cond:", sample["data_cond"])
        
        print("\nSIM CLUSTERS:", len(sample["sim_clusters"]))
        print("DATA CLUSTERS:", len(sample["data_clusters"]))

        print("\n\n\t ***** CLUSTER VARIANCE TEST *****")
        inspect_cluster_set(sample["sim_clusters"], "SIM")
        inspect_cluster_set(sample["data_clusters"], "DATA")

        print("\n\n\t ***** DATA LOADER TEST *****")
        data_loader_test(dataset)

        print("\n\n\t ***** IMAGE CLUSTER TEST *****")
        image_test(sample)
        
