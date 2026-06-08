import numpy as np
import torch
import torch.nn as nn
import ROOT
import argparse

import yaml
from yaml import Loader
import json
import hashlib

from data_reading.read_data_2D import read_data

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

    read_data(conf,args.usecache,args.onlycache)
    
    
