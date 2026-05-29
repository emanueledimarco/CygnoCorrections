import numpy as np
import torch
import torch.nn as nn
import ROOT
import argparse
import os
import uproot
import pandas as pd
from collections import defaultdict

import yaml
from yaml import Loader
import json


from data_reading.read_data_2D import build_cluster_dataframe

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--onlycache",action="store_true",help="Run only the creation of the panda dataframes and cache them")
    parser.add_argument("--configuration",type=str,default="configuration_2026_lime_v1",help="Key of the configuration in the flow yaml configuration file")
    parser.add_argument("--usecache",action="store_true",help="use cached source and target datasets in cache/ dir instead of re-reading from ROOT")
    args = parser.parse_args()

    #loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    stream = open("flow_configuration.yaml", 'r')
    dictionary = yaml.load(stream,Loader)
    conf = args.configuration

    with open("cluster_training_list.json", "r") as file:
        json_data = json.load(file)
    variables = json_data["var_scalar_list"]
    print("List of the veriables used in the flow for selection/validation:   ", variables)
    
    print(f"Now filling the datasets for the simulation and data. It applies the selection and converts them to panda DFs.  Since many files are involved, it takes time...")

    cachedir = "data/cache"
    if not args.usecache:
        source_data_lists = defaultdict(list)
        target_data_lists = defaultdict(list)
     
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
                        for rootfname in files:
                            source_data_lists[mapkey].append(build_cluster_dataframe([rootfname],variables,isdata=False))
                else:
                    print("\t==> Data now...")
                    for mapkey,files in map_dic.items():
                        for rootfname in files:
                            target_data_lists[mapkey].append(build_cluster_dataframe([rootfname],variables,isdata=True))

        # merge the PDs for the sim, which have multiple files/key
        print("Concatenate now the split SIM datasets...")
        source_data = {}
        target_data = {}
        for key,dfs in source_data_lists.items():
            print(f"\tConcatenating SIM: {len(dfs)} datasets for key {key}.") 
            source_data[key] = pd.concat(dfs, ignore_index=True)
        for key,dfs in target_data_lists.items():
            print(f"\tConcatenating DATA: {len(dfs)} datasets for key {key}.") 
            target_data[key] = pd.concat(dfs, ignore_index=True)
        
        os.makedirs(cachedir, exist_ok=True)
        pd.to_pickle(source_data, f"{cachedir}/source_data.pkl")
        pd.to_pickle(target_data, f"{cachedir}/target_data.pkl")
        print(f"Datasets selected and stored in {cachedir}")

        if args.onlycache:
            print("Exiting after caching. Now run without --onlycache")
            exit(0)

    else:
        print(f"Reading source_data and target_data from pre-selected Panda DFs in {cachedir}")
        source_data = pd.read_pickle(f"{cachedir}/source_data.pkl")
        target_data = pd.read_pickle(f"{cachedir}/target_data.pkl")

    standardize=dictionary[conf]["standardize"]

    # --- TRAINING ---- #
    if args.train: 
        pass
