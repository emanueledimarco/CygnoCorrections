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


from flow_datasets import UnpairedTransportDataset, build_val_case
from training_utils import SimulationCorrection, load_model, atomic_flow_test, print_numeric_validation, standardize_dataset, interleave
from data_reading.read_data import read_reco_data_withselection, df_to_tree
from plot.plot_utils import plot_distributions
from plot.validation_utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--onlycache",action="store_true",help="Run only the creation of the panda dataframes and cache them")
    parser.add_argument("--train",action="store_true",help="Run the training step")
    parser.add_argument("--validate",action="store_true",help="Run the validation step")
    parser.add_argument("--matrix",action="store_true",help="Run the matrix validation step (to cover all the Sim x Data conditions")
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
                            source_data_lists[mapkey].append(read_reco_data_withselection(variables,spectators,[rootfname],isdata=False))
                else:
                    print("\t==> Data now...")
                    for mapkey,files in map_dic.items():
                        for rootfname in files:
                            target_data_lists[mapkey].append(read_reco_data_withselection(variables,spectators,[rootfname],isdata=True))

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

        # prepare one "validation golden case" to define the convergence for the training
        # with periodic validation and early stopping
        # x=y=a=b=3 for example
        alphaV  = float(dictionary["data_inputs"]["alpha_ref"])
        lambdaV = float(dictionary["data_inputs"]["lambda_ref"])
        ztrueV  = float(dictionary["data_inputs"]["ztrue_ref"])
        PV      = float(dictionary["data_inputs"]["P_ref"])
        TV      = float(dictionary["data_inputs"]["T_ref"])
        HV      = float(dictionary["data_inputs"]["H_ref"])
        ZV      = ztrueV   # float(dictionary["data_inputs"]["Z_ref"])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
     
        # remove the validation case from the training datasets and add it to a separate dic
        print(f"Will use the case:\n\t(ztrue,alpha,lambda) = ({ztrueV},{alphaV},{lambdaV});\n\t(Z,P,T,H) = ({ZV},{PV},{TV},{HV})\nas the reference case to evaluate the metric during the training, so removing it from the training")

        source_key_V = (ztrueV,alphaV,lambdaV)
        if source_key_V in source_data:
            val_sim = source_data.pop(source_key_V,None)
        else:
            print(f"Warning, the element {source_key_V} is not among the simulation datasets")
     
        target_key_V = (ZV,PV,TV,HV)
        if target_key_V in target_data:
            val_data = target_data.pop(target_key_V,None)
        else:
            print(f"Warning, the element {target_key_V} is not among the data datasets")

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
        raw_context_dim = len(source_key_V) + len(target_key_V) - 1 # removed Z data
        raw_context_dim += len(variables) # add 1 latent noise to each variable in the context
        
        # build the flow and train it
        corrections = SimulationCorrection(str(conf),dictionary[conf],dataset,standardize,raw_context_dim)

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
        H0=dictionary["data_inputs"]["H_val"]
        Z0=ztrue0

        src_key_0 = (ztrue0,alpha0,lambda0)
        tgt_key_0 = (Z0,P0,T0,H0)

        # dataframe -> torch tensors conversion
        A_sim_df  = source_data[src_key_0]
        A_data_df = target_data[tgt_key_0]
        A_sim  = torch.tensor(A_sim_df.values, dtype=torch.float32, device=device)
        A_data = torch.tensor(A_data_df.values, dtype=torch.float32, device=device)

        # context construction
        src_key_0_t = torch.tensor(src_key_0, dtype=torch.float32, device=device)
        tgt_key_0_t = torch.tensor(tgt_key_0, dtype=torch.float32, device=device)
        tgt_key_0_t_reduced = tgt_key_0_t[..., 1:] # remove Z from the target context
        raw_context = torch.cat([src_key_0_t,tgt_key_0_t_reduced]).expand(A_sim.shape[0],-1)        
        sigma_latent = dictionary[conf]["sigma_latent"]
        if standardize:
            A_sim_scaled,mu_sim,std_sim = standardize_dataset(A_sim)
            A_data_scaled,mu_data,std_data = standardize_dataset(A_data)
            z_latent = sigma_latent * torch.randn_like(A_sim_scaled)
        else:
            z_latent = sigma_latent * torch.randn_like(A_sim)
        context_input = torch.cat([raw_context, z_latent], dim=1)
        
        # --- APPLICA FLOW PER LA VALIDAZIONE --- #
        print ("EVALUATE FLOW")
        
        with torch.no_grad():
            cond = context_encoder(context_input)
            A_corr_scaled, _ = flow(A_sim_scaled, cond)
            debug_variance(A_sim_scaled,raw_context,context_encoder,flow)
        
        if standardize:
            print("De-standardize A_corr")
            print(f"A_corr (scaled): mean={A_corr_scaled.mean(0)}, std={A_corr_scaled.std(0)}")
            A_corr = A_corr_scaled * std_data + mu_data
            print(f"A_corr (un-scaled): mean={A_corr.mean(0)}, std={A_corr.std(0)}")

        print("FLOW done")

        # A_corr torch → pandas with the same structure of A_sim_scaled (to plot)
        A_corr_df = pd.DataFrame(
            A_corr.detach().cpu().numpy(),
            columns=A_sim_df.columns
        )

        # validazione numerica:
        print_numeric_validation(A_sim_scaled,A_data_scaled,A_corr_scaled)
        # global metrics:
        metrics = compute_validation_metrics(A_corr_scaled,A_data_scaled)
        print("==== GLOBAL VALIDATION ====")
        for k,m in metrics.items():
            print(f"{k} : {m}")
        print("===========================")
        
        # --- CREAZIONE VALIDATOR --- #
        path_to_plots = "./plot/validation_plots/"
        plot_distributions(path_to_plots, variables, A_data_df, A_sim_df, A_corr_df, params=dictionary["data_inputs"], doratio=False)
        if len(variables)>1:
            vars_to_plot = random_ordered_pair(variables)
            plot_2d_comparison(A_sim, A_corr, A_data, vars_to_plot, path_to_plots, params=dictionary["data_inputs"])

        # --- SALVA IL ROOT FILE CON IL TREE --- #
        output_root = "validation_output.root"
        with uproot.recreate(f"{path_to_plots}/{output_root}") as f:
            df_to_tree(f, "Sim",  A_sim_df)
            df_to_tree(f, "Data", A_data_df)
            df_to_tree(f, "Corr", A_corr_df)

        print(f"ROOT validation file written to: {path_to_plots}/{output_root}")
        

    # ---  MATRIX VALIDATION --- #
    elif args.matrix:

        # --- CONFIG --- #
        checkpoint_path = os.getcwd() + "/results/" + str(conf) + "/saved_states/best_model.pt"
        device = "cpu"

        # --- CARICAMENTO MODELLI --- #
        flow, context_encoder, meta = load_model(checkpoint_path, device=device)
        print("Modello caricato!")
        print("Step migliore:", meta.get("best_step"))
        print("Val MMD:", meta.get("best_val_mmd"))
        n_matrix = len(source_data.keys())*len(target_data.keys())
        print(f"Matrix of tests: [{len(source_data.keys())}(sim)x{len(target_data.keys())}(data)] = {n_matrix} flows")
        
        metrics_list = []
        ival=-1
        for sim_k in source_data.keys():
            Z,Alpha,Lambda = sim_k
            for data_k in target_data.keys():
                # validazione e plot per casi selezionati ---
                ival += 1
                if ival%500!=0: continue
                print(f"Validating combination # {ival} ...")
                
                _,P,T,H = data_k

                src_key_0 = (Z,Alpha,Lambda)
                tgt_key_0 = (Z,P,T,H)

                # the Z is taken from sim, but it can be that the corresponding key in data is absent (not processed, not taken, etc)
                if tgt_key_0 not in target_data:
                    continue
           
           
                # dataframe -> torch tensors conversion
                A_sim_df  = source_data[src_key_0]
                A_data_df = target_data[tgt_key_0]
                A_sim  = torch.tensor(A_sim_df.values, dtype=torch.float32, device=device)
                A_data = torch.tensor(A_data_df.values, dtype=torch.float32, device=device)
                
                # context construction
                src_key_0_t = torch.tensor(src_key_0, dtype=torch.float32, device=device)
                tgt_key_0_t = torch.tensor(tgt_key_0, dtype=torch.float32, device=device)
                tgt_key_0_t_reduced = tgt_key_0_t[..., 1:] # remove Z from the target context
                raw_context = torch.cat([src_key_0_t,tgt_key_0_t_reduced]).expand(A_sim.shape[0],-1)        
                sigma_latent = dictionary[conf]["sigma_latent"]
                if standardize:
                    A_sim_scaled,mu_sim,std_sim = standardize_dataset(A_sim)
                    A_data_scaled,mu_data,std_data = standardize_dataset(A_data)
                    z_latent = sigma_latent * torch.randn_like(A_sim_scaled)
                else:
                    z_latent = sigma_latent * torch.randn_like(A_sim)
                context_input = torch.cat([raw_context, z_latent], dim=1)

                # --- APPLICA FLOW PER LA VALIDAZIONE --- #
                with torch.no_grad():
                    cond = context_encoder(context_input)
                    A_corr_scaled, _ = flow(A_sim_scaled, cond)
                           
                if standardize:
                    A_corr = A_corr_scaled * std_data + mu_data
           
           
                # A_corr torch → pandas with the same structure of A_sim_scaled (to plot)
                A_corr_df = pd.DataFrame(
                    A_corr.detach().cpu().numpy(),
                    columns=A_sim_df.columns
                )

                metrics = compute_validation_metrics(A_corr_scaled,A_data_scaled)
                metrics['case_idx'] = ival
                metrics_list.append(metrics)
                
                # --- CREAZIONE VALIDATOR --- #
                path_to_plots = "./plot/validation_plots/"
                suffix = f"z-{Z}-alpha{Alpha}-lambda{Lambda}-P{P}-T{T}-H{H}"
                params = { "ztrue_val": Z, "lambda_val": Lambda, "alpha_val": Alpha, "P_val": P, "T_val": T, "H_val": H}
                
                plot_distributions(path_to_plots, variables, A_data_df, A_sim_df, A_corr_df, params=params, doratio=False, suffix=suffix)
                if len(variables)>1:
                    vars_to_plot = random_ordered_pair(variables)
                    plot_2d_comparison(A_sim, A_corr, A_data, vars_to_plot, path_to_plots, params=params, suffix=suffix)
        
        summary = aggregate_metrics(metrics_list)
        print("==== GLOBAL VALIDATION ====")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")
        print("===========================")
                
    else:
        print("Specify at least --train or --validate or --matrix")
        exit(0)
        

