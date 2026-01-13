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

# Due to the diferences in the kinematic distirbutions of the data and MC a reweithing must be performed to account for this
def perform_reweighting(simulation_df, data_df):
    
    # Reading and normalizing the weights
    mc_weights = np.ones(len(data_df["sc_integral"]))
    mc_weights = mc_weights/np.sum( mc_weights )

    data_weights = np.ones(len(data_df["sc_integral"]))
    data_weights = data_weights/np.sum( data_weights )

    # Defining the reweigthing binning! - Bins were chossen such as each bin has ~ the same number of events
    pt_bins  = calculate_bins_position(np.array(simulation_df["probe_pt"]), 30)
    eta_bins = calculate_bins_position(np.array(simulation_df["probe_ScEta"]), 30)
    rho_bins = calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 30) #np.linspace( 5,65, 30) #calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 70)

    bins = [ pt_bins , eta_bins, rho_bins ]

    # Calculate 3D histograms
    data1 = [ np.array(simulation_df["probe_pt"]) , np.array(simulation_df["probe_ScEta"]) , np.array(simulation_df["fixedGridRhoAll"])]
    data2 = [ np.array(data_df["probe_pt"])       , np.array(data_df["probe_ScEta"])       , np.array(data_df["fixedGridRhoAll"])]

    hist1, edges = np.histogramdd(data1, bins=bins  , weights=mc_weights   , density=True)
    hist2, _     = np.histogramdd(data2, bins=edges , weights=data_weights , density=True)

    # Compute reweighing factors
    reweight_factors = np.divide(hist2, hist1, out=np.zeros_like(hist1), where=hist1!=0)

    # Find bin indices for each point in data1
    bin_indices = np.vstack([np.digitize(data1[i], bins=edges[i]) - 1 for i in range(3)]).T

    # Ensure bin indices are within valid range
    for i in range(3):
        bin_indices[:,i] = np.clip(bin_indices[:,i], 0, len(edges[i]) - 2  )
        
    # Apply reweighing factors
    simulation_weights = mc_weights * reweight_factors[bin_indices[:,0], bin_indices[:,1], bin_indices[:,2]]

    # normalizing both to one!
    data_weights       = data_weights/np.sum( data_weights )
    simulation_weights = simulation_weights/np.sum( simulation_weights )

    return data_weights, simulation_weights

def perform_cluster_selection(arrays):
    # Define individual selection criteria
    mask = (
        (arrays["sc_integral"] > 500) &
        (arrays["sc_integral"] < 10000) &
        (arrays["sc_xmean"] > 0) &
        (arrays["sc_xmean"] < 2500) &
        (arrays["sc_ymean"] > 0) &
        (arrays["sc_ymean"] < 2500)
    )

    # Apply mask to filter DataFrame
    return arrays[mask]

# This function prepare teh data to be used bu pytorch!
def separate_training_data( data_df, mc_df, input_vars, condition_vars):

    mc_inputs     = torch.tensor(np.array( np.nan_to_num(mc_df[input_vars])))
    mc_conditions = torch.tensor(np.concatenate( [  np.array( np.nan_to_num(mc_df[condition_vars])), 0*np.ones( len( mc_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    input_vars_data = input_vars
    for i in range( len(input_vars) ):
        input_vars_data[i] = input_vars_data[i].replace('_raw','')

    # creating the inputs and conditions tensors! - adding the bollean to the conditions tensors
    data_inputs     = torch.tensor(np.nan_to_num(np.array(data_df[input_vars_data])))
    data_conditions = torch.tensor(np.concatenate( [ np.nan_to_num(np.array( data_df[condition_vars])), np.ones( len( data_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    #first we shuffle all the arrays!
    permutation = np.random.permutation(len(data_weights))
    data_inputs      = torch.tensor(data_inputs[permutation])
    data_conditions  = torch.tensor(data_conditions[permutation])
    data_weights     = torch.tensor(data_weights[permutation])

    mc_permutation = np.random.permutation(len(mc_weights))
    mc_inputs      = torch.tensor(mc_inputs[mc_permutation])
    mc_conditions  = torch.tensor(mc_conditions[mc_permutation])
    mc_weights     = torch.tensor(mc_weights[mc_permutation])

    assert abs(torch.sum(mc_weights) - torch.sum(data_weights)) < 1

    # Now, in order not to bias the network we choose make sure the tensors of data and simulation have the same number of events
    try:
        mc_inputs = mc_inputs[ :len(data_inputs) ]
        mc_conditions = mc_conditions[ :len(data_inputs) ]
        mc_weights = mc_weights[ :len(data_inputs) ]
        
        assert len( mc_weights )    == len( data_weights )
        assert len( mc_conditions ) == len(data_conditions)
    except:
        data_inputs = data_inputs[ :len(mc_inputs) ]
        data_conditions = data_conditions[ :len(mc_inputs) ]
        data_weights = data_weights[ :len(mc_inputs) ]
        
        assert len( mc_weights )    == len( data_weights )
        assert len( mc_conditions ) == len(data_conditions)

    # lets normalize the weights here again just to be sure ....
    mc_weights = len(data_weights)*mc_weights/torch.sum(mc_weights)

    assert abs(torch.sum(mc_weights) - torch.sum(data_weights)) < 1

    print( 'Number of MC events after equiparing! ', len( mc_conditions ), ' Number of data events: ', len(data_conditions))

    training_percent = 0.7
    validation_percent = 0.03 + training_percent
    testing_percet = 1 - training_percent - validation_percent

    # Now, the fun part! - Separating everyhting into trainnig, validation and testing datasets!
    data_training_inputs     = data_inputs[: int( training_percent*len(data_inputs  )) ] 
    data_training_conditions = data_conditions[: int( training_percent*len(data_inputs  )) ] 
    data_training_weights    = data_weights[: int( training_percent*len(data_inputs  )) ] 

    mc_training_inputs     = mc_inputs[: int( training_percent*len(mc_inputs  )) ] 
    mc_training_conditions = mc_conditions[: int( training_percent*len(mc_inputs  )) ] 
    mc_training_weights    = mc_weights[: int( training_percent*len(mc_inputs  )) ] 

    # now, the validation dataset
    data_validation_inputs     = data_inputs[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 
    data_validation_conditions = data_conditions[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 
    data_validation_weights    = data_weights[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 

    mc_validation_inputs     = mc_inputs[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 
    mc_validation_conditions = mc_conditions[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 
    mc_validation_weights    = mc_weights[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 

    # now for the grand finalle, the test tensors
    data_test_inputs     = data_inputs[int( validation_percent*len(data_inputs  )): ] 
    data_test_conditions = data_conditions[int( validation_percent*len(data_inputs  )): ] 
    data_test_weights    = data_weights[int( validation_percent*len(data_inputs  )): ] 

    mc_test_inputs     = mc_inputs[int( validation_percent*len(mc_inputs  )): ] 
    mc_test_conditions = mc_conditions[int( validation_percent*len(mc_inputs  )):] 
    mc_test_weights    = mc_weights[int( validation_percent*len(mc_inputs  )):] 

    # now, all the tensors are saved so they can be read by the training class
    path_to_save_tensors = "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/"

    torch.save( data_training_inputs       , path_to_save_tensors + 'data_training_inputs.pt' )
    torch.save( data_training_conditions   , path_to_save_tensors + 'data_training_conditions.pt')
    torch.save( data_training_weights      , path_to_save_tensors + 'data_training_weights.pt') 

    torch.save( mc_training_inputs       , path_to_save_tensors + 'mc_training_inputs.pt' )
    torch.save( mc_training_conditions   , path_to_save_tensors + 'mc_training_conditions.pt')
    torch.save( mc_training_weights      , path_to_save_tensors + 'mc_training_weights.pt') 

    # now the validation tensors
    torch.save( data_validation_inputs       , path_to_save_tensors + 'data_validation_inputs.pt' )
    torch.save( data_validation_conditions   , path_to_save_tensors + 'data_validation_conditions.pt')
    torch.save( data_validation_weights      , path_to_save_tensors + 'data_validation_weights.pt') 

    torch.save( mc_validation_inputs       , path_to_save_tensors + 'mc_validation_inputs.pt' )
    torch.save( mc_validation_conditions   , path_to_save_tensors + 'mc_validation_conditions.pt')
    torch.save( mc_validation_weights      , path_to_save_tensors + 'mc_validation_weights.pt') 

    # now the test tensors
    torch.save( data_test_inputs       , path_to_save_tensors + 'data_test_inputs.pt' )
    torch.save( data_test_conditions   , path_to_save_tensors + 'data_test_conditions.pt')
    torch.save( data_test_weights      , path_to_save_tensors + 'data_test_weights.pt') 

    torch.save( mc_test_inputs       , path_to_save_tensors + 'mc_test_inputs.pt' )
    torch.save( mc_test_conditions   , path_to_save_tensors + 'mc_test_conditions.pt')
    torch.save( mc_test_weights      , path_to_save_tensors + 'mc_test_weights.pt')     

    return path_to_save_tensors

def read_reco_data(branches, files_data, files_mc, DokinematicsRW=False):

    data_dfs = []
    for data_file in files_data:
        with uproot.open(data_file) as f:
            tree = f["Events"]
            arrays = tree.arrays(branches, library="ak")
            arrays_sel = perform_cluster_selection(arrays)
            df = pd.DataFrame({
                v: ak.flatten(arrays_sel[v]).to_numpy()
                for v in branches
            })
            data_dfs.append(df)
    data_df = pd.concat(data_dfs,ignore_index=True)

    mc_dfs = []
    for mc_file in files_mc:
        with uproot.open(mc_file) as f:
            tree = f["Events"]
            arrays = tree.arrays(branches, library="ak")
            arrays_sel = perform_cluster_selection(arrays)
            df = pd.DataFrame({
                v: ak.flatten(arrays_sel[v]).to_numpy()
                for v in branches
            })
            mc_dfs.append(df)
    mc_df = pd.concat(mc_dfs,ignore_index=True)

    # Lets now perform a kinematic reweigthing after selection
    # if( DokinematicsRW ):
    #     data_df["weight"] ,drell_yan_df["weight"] = perform_reweighting(drell_yan_df, data_df)
    
    # Normalizing the weoghts do data!
    # data_df["weight"]      = len(data_df["weight"])*data_df["weight"]/np.sum(data_df["weight"])
    # drell_yan_df["weight"] = len(data_df["weight"])*drell_yan_df["weight"]/np.sum(drell_yan_df["weight"])
    # mc_weights_before      = len(data_df["weight"])*mc_weights_before/np.sum(mc_weights_before)
    
    # assert abs(np.sum(drell_yan_df["weight"]) - np.sum(data_df["weight"])) <= 1
    
    # # Making the weigths into numpy arrays
    # mc_weights        = drell_yan_df["weight"].values
    # data_weights      = data_df["weight"].values
    
    # assert np.sum(mc_weights ) - np.sum(data_weights) <= 1
    
    # now lets call a plotting function to perform the plots of the read distributions for validation porpuses
    path_to_plots = "./plot/validation_plots/"

    # now as a last step, we need to split the data into training, validation and test dataset
    plot_utils.plot_distributions( path_to_plots, data_df, mc_df, var_list)

    # now, the datsets will be separated into training, validation and test dataset, and saved for further reading by the training class!
    #separate_training_data(data_df, mc_df, var_list, conditions_list)

    print('\n End of data reading! - No errors encountered! ')
    print( 'Number of MC events: ', len(mc_df ), ' Number of data events: ', len(data_df))

    
if __name__ == "__main__":

    var_list = [
        "sc_xmean",
        "sc_ymean",
        "sc_integral",
        "sc_nhits",
        "sc_tgausssigma"
    ]
    
    read_reco_data(var_list,
                   ["data/runs/test_recodata/reco_run87646_3D.root"],
                   ["data/sim/test_recosim/digi_3-9/iron_step3/reco_run00003_3D.root"]
                   )
    
