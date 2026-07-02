import numpy as np
import torch
import torch.nn as nn
import ROOT
import argparse

import yaml
from yaml import Loader
import json
import hashlib

from data_reading.read_data_2D import *
from training.clusterTraining import forward_test, train_model, test_training, plot_training_history
from training.validation import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--onlycache",action="store_true",help="Run only the creation of the panda dataframes and cache them")
    parser.add_argument("--configuration",type=str,default="configuration_limerun4_v2",help="Key of the configuration in the flow yaml configuration file")
    parser.add_argument("--dscheck",action="store_true",help="do a sanity check of the conditioned dataset")   
    parser.add_argument("--integrity",action="store_true",help="use cached clusters dataset, and do integrity tests")
    parser.add_argument("--fwdtest",action="store_true",help="do the fwd test of the CYGNO transport model")
    parser.add_argument("--train",action="store_true",help="train the correction")    
    parser.add_argument("--test",action="store_true",help="train the correction")
    parser.add_argument("--validate",action="store_true",help="validate the correction by pure inference")
    parser.add_argument("--checkpoint",type=str,help="give the path of a checkpoint.pt file containing a trained model (to start the training from a given point)")
    args = parser.parse_args()

    #loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    stream = open("flow_configuration.yaml", 'r')
    dictionary = yaml.load(stream,Loader)
    conf = args.configuration

    if args.onlycache:
        read_data_and_save(conf)

    inputfile = 'data/cache/cygno_clusters_dataset.pkl'
    if args.integrity:
        integrity_tests(inputfile)

    if args.dscheck:
        dataset_sanity(inputfile)
        
    if args.fwdtest:
        forward_test(inputfile)

    outputmodel = os.getcwd() + "/results/" + str(conf) + "/saved_states/best_model.pt"
    if args.train:
        print("\n\t === TRAIN THE MODEL ===")
        startmodel=None
        if args.checkpoint:
            print(f"---> Starting the training from the pre-trained model saved in {args.checkpoint}")
        model, history = train_model(inputfile,outputmodel,args.checkpoint)
        plot_training_history(history)
        print("\n\t === TEST THE MODEL ===")
        print(f"\nTest the trained model using the saved state in {outputmodel}")
        test_training(model,inputfile)
        
    if args.test:
        test_training(outputmodel,inputfile)

    if args.validate:

        sweep_vars = ["P","T","H"]

        # -----------------------
        # 0. Device setup
        # -----------------------
        device = None
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # -----------------------
        # 1. Dataset reading and metadata (to find the keys)
        # -----------------------
        print("Data loading...")
        batch_size = 256
        _, loader = build_dataloader(inputfile, batch_size=batch_size, is_test=True, num_workers=0)

        with open(inputfile, "rb") as f:
            dataset_bundle = pickle.load(f)
        metadata = dataset_bundle.get("metadata", {})
        
        # -----------------------
        # 2. Load model
        # -----------------------
        print(f"\nLoading model from:\n{outputmodel}")
        model = CygnoTransportModel().to(device)
        state = torch.load(outputmodel, map_location=device)
        model_state = state['model_state_dict']
        current_epoch = state['epoch']
        model.load_state_dict(model_state)
        print(f"===> Loaded the best model at epoch {current_epoch}.")

        unique_z = sorted(list(set([k[0] for k in metadata["keys"]["data_keys"]])))
        print(f"====> Will make data/MC comparison for these z values: {unique_z}")

        for z_val in unique_z:
            print(f"\n--- Data/MC for z = {z_val} cm... ---")
            central_sim_key = (z_val,0.0214,1450.0) # z, alpha, lambda
            for var in sweep_vars:
                print(f"\n\t\t--> Testing variable {var} for median values of the other variables")
                run_validation_sweep_from_dict(model,metadata,loader,inputfile,central_sim_key,sweep_var=var,device=device)
