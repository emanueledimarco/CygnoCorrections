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
        model, history = train_model(inputfile,outputmodel)
        plot_training_history(history)
        print("\n\t === TEST THE MODEL ===")
        print(f"\nTest the trained model using the saved state in {outputmodel}")
        test_training(model,inputfile)
        
    if args.test:
        test_training(outputmodel,inputfile)

    if args.validate:
        with open(inputfile, "rb") as f:
            dataset_bundle = pickle.load(f)
        sim_dict = dataset_bundle["sim"]
        data_dict = dataset_bundle["data"]
        metadata = dataset_bundle.get("metadata", {})
        central_sim_key = (0.0214,1450) # alpha, lambda
        run_validation_sweep_from_dict(outputmodel,metadata,sim_dict,data_dict,sweep_var="H")
