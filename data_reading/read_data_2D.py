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

import yaml
from yaml import Loader
import json
import hashlib

from data_reading.cluster import *

selection_cfg = {
    "integral_min": 2000,
    "integral_max": 50000,
    "x_min": 500,
    "x_max": 2000,
    "y_min": 500,
    "y_max": 2000,
    "min_npix": 5
}

def read_data(conf,usecache=False,onlycache=False):

    #loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    stream = open("flow_configuration.yaml", 'r')
    dictionary = yaml.load(stream,Loader)

    with open("cluster_training_list.json", "r") as file:
        json_data = json.load(file)
        
    variables = json_data["var_scalar_list"]
    print("List of the veriables used in the flow for selection/validation:   ", variables)
    
    print(f"Now filling the datasets for the simulation and data. It applies the selection and converts them to panda DFs.  Since many files are involved, it takes time...")

    cachedir = "data/cache"
    if not usecache:
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
                        break
                else:
                    print("\t==> Data now...")
                    for mapkey,files in map_dic.items():
                        data_clusters_dict[mapkey] = build_dataset_from_files(files, variables, mapkey, isdata=True, selection_cfg=selection_cfg)
                        break

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
            "sim": ["alpha", "lambda", "z"],
            "data": ["P", "T", "z"],
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
        
        if onlycache:
            print("Exiting after caching. Now run without --onlycache")
            exit(0)

    else:
        print(f"Reading source_data and target_data from pre-selected Panda DFs in {cachedir}")
        #source_data = pd.read_pickle(f"{cachedir}/source_data.pkl")
        #target_data = pd.read_pickle(f"{cachedir}/target_data.pkl")
