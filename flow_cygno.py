import numpy as np
import torch
import torch.nn as nn
import ROOT
import argparse
import os
import uproot
import pandas as pd

import yaml
from yaml import Loader
import json


from flow_datasets import UnpairedTransportDataset, build_val_case
from training_utils import SimulationCorrection, load_model, atomic_flow_test, print_numeric_validation
from data_reading.read_data import read_reco_data_withselection, df_to_tree
from plot.plot_utils import plot_distributions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true",help="Run the training step")
    parser.add_argument("--validate",action="store_true",help="Run the validation step")
    parser.add_argument("--configuration",type=str,default="configuration_2026_lime_v1",help="Key of the configuration in the flow yaml configuration file")
    parser.add_argument("--usecache",action="store_true",help="use cached source and target datasets in cache/ dir instead of re-reading from ROOT")
    parser.add_argument("--atomictest",action="store_true",help="Do basic identity test on the training (DEBUG)")
    args = parser.parse_args()

    #loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    stream = open("flow_configuration.yaml", 'r')
    dictionary = yaml.load(stream,Loader)
    conf = args.configuration

    with open("var_training_list.json", "r") as file:
        json_data = json.load(file)
    variables = json_data["var_list"]
    spectators = json_data["spectator_list"]
    print("List of the veriables used in the flow as observables to transform:   ", variables)
    print("List of the veriables used in the flow for selection/validation:   ", spectators)
    
    print(f"Now filling the datasets for the simulation and data. It applies the selection and converts them to panda DFs.  Since many files are involved, it takes time...")

    cachedir = "data/cache"
    if not args.usecache:
        source_data = {}
        target_data = {}
     
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
                    for mapkey,rootfname in map_dic.items():
                        source_data[mapkey] = read_reco_data_withselection(variables,spectators,[rootfname],isdata=False)
                else:
                    print("\t==> Data now...")
                    for mapkey,rootfname in map_dic.items():
                        target_data[mapkey] = read_reco_data_withselection(variables,spectators,[rootfname],isdata=True)
                        
        os.makedirs(cachedir, exist_ok=True)
        pd.to_pickle(source_data, f"{cachedir}/source_data.pkl")
        pd.to_pickle(target_data, f"{cachedir}/target_data.pkl")
        print(f"Datasets selected and stored in {cachedir}")

    else:
        print(f"Reading source_data and target_data from pre-selected Panda DFs in {cachedir}")
        source_data = pd.read_pickle(f"{cachedir}/source_data.pkl")
        target_data = pd.read_pickle(f"{cachedir}/target_data.pkl")


    # --- TRAINING ---- #
    if args.train: 

        # prepare one "validation golden case" to define the convergence for the training
        # with periodic validation and early stopping
        # x=y=a=b=3 for example
        alphaV  = float(dictionary["data_inputs"]["alpha_ref"])
        lambdaV = float(dictionary["data_inputs"]["lambda_ref"])
        ztrueV  = float(dictionary["data_inputs"]["ztrue_ref"])
        PV      = float(dictionary["data_inputs"]["P_ref"])
        TV      = float(dictionary["data_inputs"]["T_ref"])
        ZV      = float(dictionary["data_inputs"]["Z_ref"])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
     
        # remove the validation case from the training datasets and add it to a separate dic
        print(f"Will use the case:\n\t(ztrue,alpha,lambda) = ({ztrueV},{alphaV},{lambdaV});\n\t(Z,P,T) = ({ZV},{PV},{TV})\nas the reference case to evaluate the metric during the training, so removing it from the training")
     
        source_key_V = (ztrueV,alphaV,lambdaV)
        if source_key_V in source_data:
            val_sim = source_data.pop(source_key_V,None)
        else:
            print(f"Warning, the element {source_key_V} is not among the simulation datasets")
     
        target_key_V = (ZV,PV,TV)
        if target_key_V in target_data:
            val_data = target_data.pop(target_key_V,None)
        else:
            print(f"Warning, the element {target_key_V} is not among the data datasets")

        standardize=dictionary[conf]["standardize"]
        dataset = UnpairedTransportDataset(
            source_data,
            target_data,
            standardize
        )

        val_case = build_val_case(
            src_key=source_key_V,
            tgt_key=target_key_V,
            source_data=val_sim,
            target_data=val_data,
            device=device
        )
        
        # context configuration
        raw_context_dim = len(source_key_V) + len(target_key_V)
        encoder_input_dim  = raw_context_dim
        encoder_hidden_dim = dictionary[conf]["encoder_hidden_dim"]
        encoder_output_dim = dictionary[conf]["encoder_output_dim"]
        encoder_n_layers   = dictionary[conf]["encoder_n_layers"]
        encoder_dropout    = dictionary[conf]["encoder_dropout"]

        # flow configuration
        flow_n_layers = dictionary[conf]["flow_n_layers"]
        flow_hidden_dim = dictionary[conf]["flow_hidden_dim"]
        flow_context_dim = dictionary[conf]["flow_context_dim"]

        # general training parameters
        initial_lr = dictionary[conf]["initial_lr"]
        batch_size = dictionary[conf]["batch_size"]
        sigma_mmd = dictionary[conf]["sigma_mmd"]
        lambda_id = dictionary[conf]["lambda_id"]

        # build the flow and train it
        corrections = SimulationCorrection(str(conf),dataset,standardize,
                                           encoder_input_dim, encoder_hidden_dim, encoder_output_dim, encoder_n_layers, encoder_dropout,
                                           flow_n_layers, flow_hidden_dim, flow_context_dim,
                                           initial_lr, batch_size, sigma_mmd, lambda_id)
        corrections.setup_flow()
        corrections.set_validation_case(val_case)
        corrections.train_the_flow(test_identity=args.atomictest)

        if args.atomictest:
            atomic_flow_test(
                corrections.flow,
                dim_x=corrections.flow.dim,
                dim_c=corrections.context_encoder.output_dim,
                device=device
        )
        
        
    # ---  VALIDATION --- #
    elif args.validate:

        # --- CONFIG --- #
        checkpoint_path = os.getcwd() + "/results/" + str(conf) + "/saved_states/best_model.pt"
        device = "cpu"

        # --- CARICAMENTO MODELLI --- #
        flow, context_encoder, meta = load_model(checkpoint_path, device=device)
        print("Modello caricato!")
        print("Step migliore:", meta.get("best_step"))
        print("Val MMD:", meta.get("best_val_mmd"))

        # --- ESEMPIO: generiamo un caso di validazione per una coppia (x,y) ---
        # seleziona uno xy di validazione
        # Nominal simulation, e.g. "central pair" of sim parameters x0,y0
        # observed environmental parameters (example within the training range) a0,b0
        alpha0=dictionary["data_inputs"]["alpha_val"]
        lambda0=dictionary["data_inputs"]["lambda_val"]
        ztrue0=dictionary["data_inputs"]["ztrue_val"]
        P0=dictionary["data_inputs"]["P_val"]
        T0=dictionary["data_inputs"]["T_val"]
        Z0=dictionary["data_inputs"]["Z_val"]

        src_key_0 = (ztrue0,alpha0,lambda0)
        tgt_key_0 = (Z0,P0,T0)
        
        # context construction
        src_key_0_t = torch.tensor(src_key_0, dtype=torch.float32, device=device)
        tgt_key_0_t = torch.tensor(tgt_key_0, dtype=torch.float32, device=device)
        context = torch.cat([src_key_0_t,tgt_key_0_t]).unsqueeze(0)

        # dataframe -> torch tensors conversion
        A_sim_df  = source_data[src_key_0]
        A_data_df = target_data[tgt_key_0]
        A_sim  = torch.tensor(A_sim_df.values, dtype=torch.float32, device=device)
        A_data = torch.tensor(A_data_df.values, dtype=torch.float32, device=device)

        cond = context_encoder(context)

        # --- APPLICA FLOW PER LA VALIDAZIONE --- #
        print ("EVALUATE FLOW")
        # replica il context per ogni evento di A_sim
        context_rep = context.repeat(A_sim.shape[0], 1)
        with torch.no_grad():
            cond = context_encoder(context_rep)
            A_corr, _ = flow(A_sim, cond)
        print("FLOW done")

        # A_corr torch → pandas with the same structure of A_sim (to plot)
        A_corr_df = pd.DataFrame(
            A_corr.detach().cpu().numpy(),
            columns=A_sim_df.columns
        )

        # validazione numerica:
        print_numeric_validation(A_sim,A_data,A_corr)

        
        # --- CREAZIONE VALIDATOR --- #
        path_to_plots = "./plot/validation_plots/"
        plot_distributions(path_to_plots, variables, A_data_df, A_sim_df, A_corr_df)

        # --- SALVA IL ROOT FILE CON IL TREE --- #
        output_root = "validation_output.root"
        with uproot.recreate(f"{path_to_plots}/{output_root}") as f:
            df_to_tree(f, "Sim",  A_sim_df)
            df_to_tree(f, "Data", A_data_df)
            df_to_tree(f, "Corr", A_corr_df)

        print(f"ROOT validation file written to: {path_to_plots}/{output_root}")
        
    else:
        print("Specify at least --train or --validate")
        exit(0)
        
