# Loads and treat the reconstruction tree data and simulation
# test basic plotting with python -m data_reading.read_data 

# importing libraries
import os 
import glob
import torch
import uproot
import pandas as pd
import awkward as ak
import numpy as np

# importing other scripts
from plot import plot_utils

def calculate_bins_position(array, num_bins=12):

    array_sorted = np.sort(array)  # Ensure the array is sorted
    n = len(array)
    
    # Calculate the exact number of elements per bin
    elements_per_bin = n // num_bins
    
    # Adjust bin_indices to accommodate for numpy's 0-indexing and avoid out-of-bounds access
    bin_indices = [i*elements_per_bin for i in range(1, num_bins)]
    
    # Find the array values at these adjusted indices
    bin_edges = array_sorted[bin_indices]

    bin_edges = np.insert(bin_edges, 0, np.min(array))
    bin_edges = np.append(bin_edges, np.max(array))
    
    return bin_edges

def perform_cluster_selection(arrays,isdata):
    if isdata: 
        # Define individual selection criteria
        # basically remove noise at low LY and borders of the sensor
        mask = (
            (arrays["sc_integral"] > 2000) &
            (arrays["sc_integral"] < 10000) &
            (arrays["sc_xmean"] > 500) &
            (arrays["sc_xmean"] < 2000) &
            (arrays["sc_ymean"] > 500) &
            (arrays["sc_ymean"] < 2000)
        )

        # Apply mask to filter DataFrame
        return arrays[mask]
    else:
        # no noise, no border effects in sim (for now)
        return arrays

def read_reco_data_withselection(variables, spectators, rootfiles, isdata, return_spectators=False, verbose=True):
    branches = variables+spectators
    data_dfs = []
    for data_file in rootfiles:
        if verbose:
            print (f"Will open the rootfile {data_file}")
        with uproot.open(data_file) as f:
            tree = f["Events"]
            arrays = tree.arrays(branches, library="ak")
            arrays_sel = perform_cluster_selection(arrays,isdata)
            df = pd.DataFrame({
                v: ak.flatten(arrays_sel[v]).to_numpy()
                for v in branches
            })
            data_dfs.append(df)
    data_df = pd.concat(data_dfs,ignore_index=True)
    if not return_spectators:
        return data_df[variables]
    else:
        return data_df

def validate_data_and_mc(variables, spectators, files_data, files_mc):

    data_df = read_reco_data_withselection(branches, files_data)
    mc_df = read_reco_data_withselection(branches, files_mc)
    
    # now lets call a plotting function to per form the plots of the read distributions for validation porpuses
    path_to_plots = "./plot/validation_plots/"

    # now as a last step, we need to split the data into training, validation and test dataset
    plot_utils.plot_distributions( path_to_plots, data_df, mc_df, var_list)

    # now, the datsets will be separated into training, validation and test dataset, and saved for further reading by the training class!
    #separate_training_data(data_df, mc_df, var_list, conditions_list)

    print('\n End of data reading! - No errors encountered! ')
    print( 'Number of MC events: ', len(mc_df ), ' Number of data events: ', len(data_df))

    return mc_df, data_df

class TreeFlowDataset(torch.utils.data.Dataset):
    def __init__(self, sim_df, data_df, var_list, context):
        # numpy conversion
        A_sim = sim_df[var_list].to_numpy(dtype=np.float32)  # shape (N_sim, D)
        A_data = data_df[var_list].to_numpy(dtype=np.float32)  # shape (N_data, D)

        # conversion to pytorch
        """
        A_sim:  (N_sim, D)
        A_data: (N_data, D)
        context: (N_context, C)
        """
        self.A_sim = torch.tensor(A_sim, dtype=torch.float32)
        self.A_data = torch.tensor(A_data, dtype=torch.float32)
        self.context = torch.tensor(context, dtype=torch.float32)

    def __len__(self):
        return min(len(self.A_sim), len(self.A_data))

    def __getitem__(self, idx):
        return self.A_sim[idx], self.A_data[idx], self.context[idx]

def df_to_tree(root_file, tree_name, df):
    root_file.mktree(tree_name,
                     {
                         col: df[col].to_numpy(dtype=np.float32)
                         for col in df.columns
                     }
                     )


if __name__ == "__main__":

    var_list = [
        "sc_xmean",
        "sc_ymean",
        "sc_integral",
        "sc_nhits",
        "sc_tgausssigma"
    ]
    
    validate_data_and_mc(var_list,[],
                         ["data/runs/test_recodata/reco_run95744_3D.root"],
                         ["data/sim/test_recosim/digi_3-6/iron_step4/reco_run00002_3D.root"]
                         )
    
