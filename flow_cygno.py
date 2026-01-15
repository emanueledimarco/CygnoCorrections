import numpy as np
import torch
import torch.nn as nn
import ROOT
import argparse
import os
import pandas as pd

import yaml
from yaml import Loader
import json


from flow_datasets import UnpairedTransportDataset, build_val_case
from training_utils import SimulationCorrection, load_model
from data_reading.read_data import read_reco_data_withselection

def pair_grid(x,y,Ny):
    return x * Ny + y

def plot_dataset(DS,pname="A_out"):

    h_out = ROOT.TH2D("out", "out",
                      100, -3, 7,
                      100, -5, 5)
    
    A = DS.cpu().numpy()
    for x, y in A:
        h_out.Fill(x, y)

    c = ROOT.TCanvas("c","",600,600)
    h_out.Draw("colz")
    c.SaveAs(f"{pname}.pdf")



def save_arrays_root(fname, A_sim, A_corr, A_data):
    f = ROOT.TFile(fname, "RECREATE")
    t = ROOT.TTree("events", "Flow validation")

    import array
    xs = array.array('f', [0])
    ys = array.array('f', [0])
    xc = array.array('f', [0])
    yc = array.array('f', [0])
    xd = array.array('f', [0])
    yd = array.array('f', [0])

    t.Branch("sim_x", xs, "sim_x/F")
    t.Branch("sim_y", ys, "sim_y/F")
    t.Branch("corr_x", xc, "corr_x/F")
    t.Branch("corr_y", yc, "corr_y/F")
    t.Branch("data_x", xd, "data_x/F")
    t.Branch("data_y", yd, "data_y/F")

    for i in range(len(A_sim)):
        xs[0], ys[0] = A_sim[i]
        xc[0], yc[0] = A_corr[i]
        if i < len(A_data):
            xd[0], yd[0] = A_data[i]
        t.Fill()

    t.Write()
    f.Close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true",help="Run the training step")
    parser.add_argument("--validate",action="store_true",help="Run the validation step")
    parser.add_argument("--configuration",type=str,default="configuration_2026_lime_v1",help="Key of the configuration in the flow yaml configuration file")
    parser.add_argument("--usecache",action="store_true",help="use cached source and target datasets in cache/ dir instead of re-reading from ROOT")
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
    
    z = 15.0 # for the moment, one training for a test z. Next step is to make z another condition of the flow

    print(f"Now filling the datasets for the simulation and data, for z={z}. It applies the selection and converts them to panda DFs.  Since many files are involved, it takes time...")

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
                        source_data[mapkey] = read_reco_data_withselection(variables,spectators,[rootfname])
                else:
                    print("\t==> Data now...")
                    for mapkey,rootfname in map_dic.items():
                        target_data[mapkey] = read_reco_data_withselection(variables,spectators,[rootfname])
                        
        os.makedirs(cachedir, exist_ok=True)
        pd.to_pickle(source_data, f"{cachedir}/source_data.pkl")
        pd.to_pickle(target_data, f"{cachedir}/target_data.pkl")
        print(f"Datasets selected and stored in {cachedir}")

    else:
        print(f"Reading source_data and target_data from pre-selected Panda DFs in {cachedir}")
        source_data = pd.read_pickle(f"{cachedir}/source_data.pkl")
        target_data = pd.read_pickle(f"{cachedir}/target_data.pkl")

    # prepare one "validation golden case" to define the convergence for the training
    # with periodic validation and early stopping
    # x=y=a=b=3 for example
    alphaV  = float(dictionary["data_inputs"]["alpha_ref"])
    lambdaV = float(dictionary["data_inputs"]["lambda_ref"])
    PV      = float(dictionary["data_inputs"]["P_ref"])
    TV      = float(dictionary["data_inputs"]["T_ref"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # remove the validation case from the training datasets and add it to a separate dic
    print(f"Will use the case:\n\tz = {z};\n\t(alpha,lambda) = ({alphaV},{lambdaV});\n\t(P,T) = ({PV},{TV})\nas the reference case to evaluate the metric during the training, so removing it from the training")

    source_key_V = (z,alphaV,lambdaV)
    if source_key_V in source_data:
        val_sim = source_data.pop(source_key_V,None)
    else:
        print(f"Warning, the element {source_key_V} is not among the simulation datasets")

    target_key_V = (z,PV,TV)
    if target_key_V in target_data:
        val_data = target_data.pop(target_key_V,None)
    else:
        print(f"Warning, the element {target_key_V} is not among the data datasets")

    dataset = UnpairedTransportDataset(
        source_data,
        target_data,
        device=device
    )

    # --- TRAINING ---- #
    if args.train: 

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
        corrections = SimulationCorrection(str(conf),dataset,
                                           encoder_input_dim, encoder_hidden_dim, encoder_output_dim, encoder_n_layers, encoder_dropout,
                                           flow_n_layers, flow_hidden_dim, flow_context_dim,
                                           initial_lr, batch_size, sigma_mmd, lambda_id)
        corrections.setup_flow()
        corrections.set_validation_case(val_case)
        corrections.train_the_flow()
        
     
        
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
        x0=dictionary["data_inputs"]["x_val"]
        y0=dictionary["data_inputs"]["y_val"]
        a0=dictionary["data_inputs"]["a_val"]
        b0=dictionary["data_inputs"]["b_val"]

        A_sim = sample_from_th2(source_hists[(x0,y0)],n_samples=50000)
        A_data = sample_from_th2(target_hists[a0,b0],n_samples=50000)
        
        # seleziona un caso di context per il flow
        context_values = [x0, y0, a0, b0]
        context_tensor = torch.tensor([context_values]*len(A_sim), dtype=torch.float32)
        cond = context_encoder(context_tensor)

        # --- APPLICA FLOW PER LA VALIDAZIONE --- #
        print ("EVALUATE FLOW")
        A_corr = flow(A_sim.detach().clone(), cond)
        print("FLOW done")

        # --- CREAZIONE VALIDATOR --- #
        validator = TH2FlowValidation(A_sim=A_sim,
                                      A_corr=A_corr,
                                      A_data=A_data,
                                      xmin=-3,xmax=7,
                                      ymin=-5,ymax=5)

        # --- PRODUZIONE PLOT MULTIPAGINA PDF --- #
        pdf_name = "validation_multipage.pdf"
        validator.plot_validation(pdf_name)
        print(f"Validazione completata. File PDF generato: {pdf_name}")

        # --- SALVA IL ROOT FILE CON IL TREE --- #
        save_arrays_root("trained_tree_valid.root", A_sim, A_corr, A_data)
        
        
    else:
        print("Specify at least --train or --validate")
        exit(0)
        
