# script made to plot the main validation and resulting distributions

# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
import os

# Names of the used variables, I copied it here only so it is easier to use it acess the labels and names of teh distirbutions
var_list = ["sc_integral",
            "sc_tgausssigma",
            "sc_nhits"]

# The next three functions are related to the plotting of the profiles of the LY as a function of cluster shape variables
# This function calculate the means of the input quantiles! - Weighted mean, of couse.
def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

# this is the main plotting function, all the other will basically set up something to call this one in the end!
def plott(data_hist,mc_hist,mc_rw_hist ,output_filename,xlabel ):

    plt.close()
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    # Check if ax[0] is indeed a matplotlib Axes object
    if not isinstance(ax[0], plt.Axes):
        raise ValueError("ax[0] must be a matplotlib Axes object")
        
    hep.histplot(
                mc_hist,
                density=True,
                label = r'Simulation',
                color = "blue",
                linewidth=3,
                ax=ax[0],
                flow='sum'
            )

    hep.histplot(
                mc_rw_hist,
                density=True,
                label=r'Simulation (Corr)',
                color = "None",
                linewidth=3,
                linestyle='--',
                ax=ax[0],
                flow='sum',
                histtype='fill',
                hatch='//',
                edgecolor='green',
                
            )

    hep.histplot(
            data_hist,
            density=True,
            label = r'Data',
            yerr=True,
            xerr=False,
            color="black",
            #linewidth=3,
            histtype='errorbar',
            markersize=12,
            elinewidth=3,
            ax=ax[0],
            flow='sum'
        )

    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.15*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)

    # Log scale for Iso variables
    if( "Iso" in str(xlabel) or "DR" in str(xlabel) or "esE" in str(xlabel)  ): # or 'r9' in str(xlabel) or 's4' in str(xlabel)
        ax[0].set_yscale('log')
        #ax[0].set_ylim(0.001,( np.max(data_hist)/1.5e6 ))
        ax[0].set_ylim(0.001, 12.05*ax[0].get_ylim()[1])
        
    # Line at 1
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=1)#, alpha=0.5)

    data_hist_numpy = data_hist.to_numpy()
    mc_hist_numpy   = mc_hist.to_numpy()
    mc_hist_rw_numpy   = mc_rw_hist.to_numpy()

    integral_data = data_hist.sum() * (data_hist_numpy[1][1] - data_hist_numpy[1][0])
    integral_mc = mc_hist.sum() * (mc_hist_numpy[1][1] - mc_hist_numpy[1][0])

    # Ratio between normalizng flows prediction and data
    ratio = (data_hist_numpy[0] / integral_data) / ( (mc_hist_numpy[0] + 1e-15 ) / integral_mc)
    ratio = np.nan_to_num(ratio)

    integral_mc_rw = mc_rw_hist.sum() * (mc_hist_rw_numpy[1][1] - mc_hist_rw_numpy[1][0])
    ratio_rw = (data_hist_numpy[0] / integral_data) / ( (mc_hist_rw_numpy[0] +1e-15 ) / integral_mc_rw)
    ratio_rw = np.nan_to_num(ratio_rw)

    errors_nom = (np.sqrt(data_hist_numpy[0])/integral_data) / ( (mc_hist_numpy[0] + 1e-15 ) / integral_mc)
    errors_nom = np.abs(np.nan_to_num(errors_nom))

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="blue",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1],
        xerr=True,
    )

    hep.histplot(
        ratio_rw,
        bins=data_hist_numpy[1],
        label=None,
        color="green",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1],
        xerr=True,
    )

    bin_width = round(data_hist.axes[0].edges[1] - data_hist.axes[0].edges[0],2)
    if "hoe" in str(xlabel):
        bin_width = round(data_hist.axes[0].edges[1] - data_hist.axes[0].edges[0],3)
    
    if( "Err" in str(xlabel) ):
        ax[0].set_ylabel("a.u", fontsize=30)
    else:
        ax[0].set_ylabel("a.u", fontsize=30)
    
    ax[1].set_ylabel("Data / MC", fontsize=26)
    
    if( "integral" in str(xlabel)  ):
        xlabel_new = "light integral [counts]"
        ax[1].set_xlabel( str(xlabel_new) , fontsize=26)
    elif( "tgausssigma" in str(xlabel)  ):
        xlabel_new = r'\sigma_{t} [pix]'
        ax[1].set_xlabel( str(xlabel_new) , fontsize=26)
    elif( "nhits" in str(xlabel)  ):
        xlabel_new = r'n_{hits}'
        ax[1].set_xlabel( str(xlabel_new) , fontsize=26)
    else:
        xlabel_new = xlabel.replace("sc_", "")
        ax[1].set_xlabel( str(xlabel_new) , fontsize=26)
    
    ax[0].tick_params(labelsize=24)

    ax[1].set_ylim(0.79, 1.21)
    if( 'integral' in xlabel ):
        ax[1].set_ylim(0, 1e4)

    # Create a custom legend handle to show a line
    from matplotlib.lines import Line2D
    line = Line2D([0], [0], color='blue', linewidth=3)

    # Get existing legend handles and labels
    handles, labels = ax[0].get_legend_handles_labels()

    # Replace the handle for the first histogram with the custom line
    handles[1] = line

    # Add the legend with the modified handles
    ax[0].legend(handles=handles, labels=labels, loc="upper right", fontsize=20)
    ax[0].text(0.05, 0.96, r'$^{55}Fe$ data/sim', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')

    hep.cms.label(data=True, ax=ax[0], loc=0, label = "Preliminary", com=13.6, lumi = 27.24)

    # Remove the space between the subplots
    plt.subplots_adjust(hspace=0)

    ax[0].margins(x=0)
    ax[1].margins(x=0)

    # Adjust the tight_layout to not add extra padding
    fig.tight_layout(h_pad=0, w_pad=0)

    fig.savefig(output_filename)

def plot_distributions( path, data_df, mc_df, variables_to_plot, mc_weights=None ):

    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    for variable in variables_to_plot:

        # Extract data for the variable
        data_values = data_df[variable].dropna().values
        mc_values = mc_df[variable].dropna().values

        # Compute mean and standard deviation for data histogram binning
        mean = np.mean(data_values)
        std = np.std(data_values)

        # Define histogram bin edges based on the variable name
        if 'Iso' in variable or 'DR' in variable:
            bin_edges = np.linspace(0.0, 3.5, 101)
        elif 'hoe' in variable:
            bin_edges = np.linspace(0.0, 0.08, 101)
        else:
            bin_min = mean - 3.0 * std
            bin_max = mean + 4.0 * std
            bin_edges = np.linspace(bin_min, bin_max, 101)

        # Create histograms
        bins = hist.axis.Variable(bin_edges)
        data_hist = hist.Hist(bins)
        mc_hist = hist.Hist(bins)
        mc_rw_hist = hist.Hist(bins)

        # Fill histograms
        data_hist.fill(data_values)
        mc_hist.fill(mc_values)
        if mc_weights is not None:
            mc_rw_hist.fill(mc_values, weight=mc_weights)
        else:
            mc_rw_hist.fill(mc_values)

        # Plot and save the histograms
        output_path = os.path.join(path, f"{variable}.png")
        plott(data_hist, mc_hist, mc_rw_hist, output_path, xlabel=variable)

def plot_distributions_for_tensors(
    data_tensor,
    mc_tensor,
    flow_samples,
    mc_weights,
    plot_path,
    variables_list
):

    # Ensure the output directory exists
    os.makedirs(plot_path, exist_ok=True)

    # Convert tensors to NumPy arrays if they are PyTorch tensors
    if isinstance(data_tensor, torch.Tensor):
        data_tensor = data_tensor.cpu().numpy()
    if isinstance(mc_tensor, torch.Tensor):
        mc_tensor = mc_tensor.cpu().numpy()
    if isinstance(flow_samples, torch.Tensor):
        flow_samples = flow_samples.cpu().numpy()
    if isinstance(mc_weights, torch.Tensor):
        mc_weights = mc_weights.cpu().numpy()

    n_features = data_tensor.shape[1]
    n_variables = len(variables_list)
    if n_features != n_variables:
        raise ValueError("Number of features in data_tensor does not match length of variables_list.")

    for i in range(n_features):
        variable_name = variables_list[i]

        # Extract data for the variable
        data_values = data_tensor[:, i]
        mc_values = mc_tensor[:, i]
        flow_values = flow_samples[:, i]

        # Compute mean and standard deviation for data histogram binning
        mean = np.mean(data_values)
        std = np.std(data_values)

        # Determine histogram binning based on variable name
        if any(substring in variable_name for substring in ['Iso', 'DR', 'esE', 'hoe', 'energy']):
            bin_min = 0.0
            bin_max = mean + 2.0 * std
            # Special case for 'DR04'
            if 'DR04' in variable_name:
                bin_max = 5.0
            bins = hist.axis.Regular(50, bin_min, bin_max)
        else:
            bin_min = mean - 2.5 * std
            bin_max = mean + 2.5 * std
            bins = hist.axis.Regular(50, bin_min, bin_max)

        # Create histograms
        data_hist = hist.Hist(bins)
        mc_hist = hist.Hist(bins)
        mc_rw_hist = hist.Hist(bins)

        # Fill histograms
        data_hist.fill(data_values)
        mc_hist.fill(mc_values, weight=mc_weights)
        mc_rw_hist.fill(flow_values, weight=mc_weights)

        # Plot and save histograms
        output_file = os.path.join(plot_path, f"{variable_name}.png")
        plott(data_hist, mc_hist, mc_rw_hist, output_file, xlabel=variable_name)

def plot_loss_cruve(training,validation, plot_path):

        fig, ax1 = plt.subplots()

        # Plot training loss on the first axis
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(training, color=color, marker='o', label='Training Loss')
        ax1.plot(validation, color='tab:orange', marker='x', label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend()

        # Title and show the plot
        plt.title('Training Loss and MVA_chsqrd')

        plt.savefig( plot_path + 'loss_plot.png') 
        plt.close()

def plot_distributions_after_transformations(
    var_list,
    training_inputs,
    conditions_list,
    training_conditions,
    training_weights,
    output_path='plot/validation_plots/after_transformation/'
):
    """
    Plots distributions of variables and conditions after transformations.

    Parameters:
    - var_list (list of str): List of variable names corresponding to training_inputs columns.
    - training_inputs (numpy.ndarray or torch.Tensor): Array of input variables (samples x features).
    - conditions_list (list of str): List of condition names corresponding to training_conditions columns.
    - training_conditions (numpy.ndarray or torch.Tensor): Array of condition variables (samples x conditions).
    - training_weights (numpy.ndarray or torch.Tensor): Array of training weights for each sample.
    - output_path (str, optional): Directory where plots will be saved.
      Defaults to 'plot/validation_plots/after_transformation/'.
    """

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # lets remove energy raw from here if it exists!
    #var_list = [entry for entry in var_list if "probe_energyRaw" not in entry.lower() and "probe_energyRaw" not in entry.lower()]
    #var_list = var_list[1:] # hotfix to elimnate energyRaw from the inputs


    # Convert inputs to NumPy arrays if they are PyTorch tensors
    if isinstance(training_inputs, torch.Tensor):
        training_inputs = training_inputs.cpu().numpy()
    if isinstance(training_conditions, torch.Tensor):
        training_conditions = training_conditions.cpu().numpy()
    if isinstance(training_weights, torch.Tensor):
        training_weights = training_weights.cpu().numpy()

    # Masks to separate events between data and MC based on the last column of training_conditions
    data_mask = training_conditions[:, -1] == 1
    mc_mask = training_conditions[:, -1] == 0

    # Ensure that masks are boolean NumPy arrays
    data_mask = data_mask.astype(bool)
    mc_mask = mc_mask.astype(bool)

    # Plot distributions for input variables
    num_variables = training_inputs.shape[1]
    for i in range(num_variables):
        variable_name = var_list[i]

        # Extract data for the variable
        data_values = training_inputs[data_mask, i]
        mc_values = training_inputs[mc_mask, i]
        mc_weights = training_weights[mc_mask]

        # Compute mean and standard deviation for data histogram binning
        mean = np.mean(data_values)
        std = np.std(data_values)

        # Create histograms with appropriate binning
        #bins = hist.axis.Regular(70, mean - 3.0 * std, mean + 3.0 * std)
        bins = hist.axis.Regular(70, -5.0, 5.0)
        bins = hist.axis.Regular(70, np.min(data_values) , np.max(data_values))
        data_hist = hist.Hist(bins)
        mc_hist = hist.Hist(bins)

        # Fill histograms
        data_hist.fill(data_values)
        mc_hist.fill(mc_values, weight=mc_weights)

        # Plot and save histograms
        output_file = os.path.join(output_path, f'after_transform_{variable_name}.png')
        plott(data_hist, mc_hist, mc_hist, output_file, xlabel=variable_name)

    # Plot distributions for condition variables
    num_conditions = training_conditions.shape[1] - 1
    for i in range(num_conditions):
        condition_name = conditions_list[i]

        # Extract data for the condition
        data_values = training_conditions[data_mask, i]
        mc_values = training_conditions[mc_mask, i]
        mc_weights = training_weights[mc_mask]

        # Compute mean and standard deviation for data histogram binning
        mean = np.mean(data_values)
        std = np.std(data_values)

        # Create histograms with appropriate binning
        bins = hist.axis.Regular(70, mean - 3.0 * std, mean + 3.0 * std)
        data_hist = hist.Hist(bins)
        mc_hist = hist.Hist(bins)

        # Fill histograms
        data_hist.fill(data_values)
        mc_hist.fill(mc_values, weight=mc_weights)

        # Plot and save histograms
        output_file = os.path.join(output_path, f'after_transform_{condition_name}.png')
        plott(data_hist, mc_hist, mc_hist, output_file, xlabel=condition_name)

# Converting covariance to correlation matrices
def cov_to_corr(cov_matrix):
    stddev = torch.sqrt(torch.diag(cov_matrix))
    stddev_matrix = torch.diag_embed(stddev)
    corr_matrix = torch.inverse(stddev_matrix) @ cov_matrix @ torch.inverse(stddev_matrix)
    return corr_matrix

def plot_correlation_matrix_diference_barrel(
    var_list_barrel_only, 
    match_indices_barrel,
    data, 
    data_conditions,
    data_weights,
    mc, 
    mc_conditions,
    mc_weights,
    mc_corrected,
    path
):

    # Remove 'probe_' prefix from variable names
    var_list_barrel_only = [var.replace('probe_', '').replace('Cone', '').replace('trk', '').replace('Pt', '').replace('Charged', '').replace('Cluster', '') for var in var_list_barrel_only]

    # Apply barrel-only condition (|eta| < 1.4222)
    barrel_eta_cut = 1.4222
    mask_data = np.abs(data_conditions[:, 1]) < barrel_eta_cut
    mask_mc = np.abs(mc_conditions[:, 1]) < barrel_eta_cut

    # Apply the barrel condition to the data
    data = data[mask_data]
    data_weights = data_weights[mask_data]

    # Apply the barrel condition to the MC samples
    mc = mc[mask_mc]
    mc_weights = mc_weights[mask_mc]
    mc_corrected = mc_corrected[mask_mc]

    # Select only the variables related to barrel variables
    data = data[:, match_indices_barrel]
    mc = mc[:, match_indices_barrel]
    mc_corrected = mc_corrected[:, match_indices_barrel]

    # Handle negative weights by taking absolute values
    data_weights_abs = np.abs(data_weights)
    mc_weights_abs   = np.abs(mc_weights)

    #calculating the covariance matrix of the pytorch tensors
    """ 
    data_cov         = torch.cov( data.T  )
    mc_cov           = torch.cov( mc.T           , aweights = torch.Tensor(abs(mc_weights_abs)))
    mc_corrected_cov = torch.cov( mc_corrected.T , aweights = torch.Tensor(abs(mc_weights_abs)))

    data_corr         = cov_to_corr(data_cov)
    mc_corr           = cov_to_corr(mc_cov)
    mc_corrected_corr = cov_to_corr(mc_corrected_cov)
    """ 

    
    # Function to compute weighted covariance matrix
    def weighted_covariance(X, weights):
        average = np.average(X, axis=0, weights=weights)
        X_centered = X - average
        cov = np.cov(X_centered, rowvar=False, aweights=weights)
        return cov

    # Compute weighted covariance matrices
    data_cov = weighted_covariance(data, data_weights_abs)
    mc_cov = weighted_covariance(mc, mc_weights_abs)
    mc_corrected_cov = weighted_covariance(mc_corrected, mc_weights_abs)

    # Convert covariance matrices to correlation matrices
    def covariance_to_correlation(cov):
        diag = np.sqrt(np.diag(cov))
        outer_diag = np.outer(diag, diag)
        corr = cov / outer_diag
        corr[cov == 0] = 0
        return corr

    data_corr = covariance_to_correlation(data_cov)
    mc_corr = covariance_to_correlation(mc_cov)
    mc_corrected_corr = covariance_to_correlation(mc_corrected_cov)

    # Helper function to plot difference matrices
    def plot_difference_matrix(diff_matrix, variable_names, xlabel, output_filename):
        fig, ax = plt.subplots(figsize=(20, 20))
        cax = ax.matshow(100 * diff_matrix, cmap='bwr', vmin=-5, vmax=5)
        fig.colorbar(cax)

        mean_diff = np.mean(np.abs(diff_matrix)) * 100

        for (i, j), z in np.ndenumerate(100 * diff_matrix):
            if abs(z) >= 1:
                ax.text(j, i, f'{z:.1f}', ha='center', va='center', fontsize=20)

        ax.set_xticks(np.arange(len(variable_names)))
        ax.set_yticks(np.arange(len(variable_names)))
        ax.set_xticklabels(variable_names, fontsize=25, rotation=90)
        ax.set_yticklabels(variable_names, fontsize=25)
        ax.set_xlabel(f'{xlabel} - Metric: {mean_diff:.2f}', loc = 'center', fontsize=30)

        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

    # Plotting difference between data and corrected MC correlation matrices
    diff_corr_corrected = data_corr - mc_corrected_corr
    plot_difference_matrix(
        diff_matrix=diff_corr_corrected,
        variable_names=var_list_barrel_only,
        xlabel='(Corr_MC_Corrected - Corr_Data)',
        output_filename=f'{path}/correlation_matrix_corrected_barrel.png'
    )

    # Plotting difference between data and uncorrected MC correlation matrices
    diff_corr = data_corr - mc_corr
    plot_difference_matrix(
        diff_matrix=diff_corr,
        variable_names=var_list_barrel_only,
        xlabel='(Corr_MC - Corr_Data)',
        output_filename=f'{path}/correlation_matrix_barrel.png'
    )

def plot_correlation_matrix_diference_endcap(data, data_conditions,data_weights,mc , mc_conditions,mc_weights,mc_corrected,path):

    # Selecting only end-cap events
    mask_mc,mask_data = np.abs(mc_conditions[:,1]) > 1.56, np.abs(data_conditions[:,1]) > 1.56

    # apply the barrel only condition
    data, mc, mc_corrected              = data[mask_data]            , mc[mask_mc]             ,mc_corrected[mask_mc]
    data_conditions,mc_conditions      = data_conditions[mask_data] , mc_conditions[mask_mc]
    data_weights,mc_weights            = data_weights[mask_data]    , mc_weights[mask_mc]
    
    # removing energy Raw from here!
    data          = data[:,1:]
    mc            = mc[:,1:]
    mc_corrected  = mc_corrected[:,1:]   

    # Some weights can of course be negative, so I had to use the abs here, since it does not accept negative weights ...
    data_corr         = torch.cov( data.T         , aweights = torch.Tensor( abs(data_weights) ))
    mc_corr           = torch.cov( mc.T           , aweights = torch.Tensor( abs(mc_weights)   ))
    mc_corrected_corr = torch.cov( mc_corrected.T , aweights = torch.Tensor( abs(mc_weights)   ))

    #from covariance to correlation matrices
    data_corr         = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(data_corr))) ) @ data_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(data_corr))) ) 
    mc_corr           = torch.inverse( torch.sqrt( torch.diag_embed(torch.diag(mc_corr)) )) @ mc_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corr))) ) 
    mc_corrected_corr = torch.inverse( torch.diag_embed(torch.sqrt(torch.diag(mc_corrected_corr)) )) @ mc_corrected_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corrected_corr))) ) 
    # end of matrix evaluations, now the plotting part!

    #plloting part
    fig, ax = plt.subplots(figsize=(33,33))
    ax.matshow( 100*( data_corr - mc_corrected_corr ), cmap = 'bwr', vmin=-5, vmax=5)

    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corrected_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 60)    
    
    mean = mean/count
    ax.set_xlabel(r'(Corr_MC$^{Corrected}$-Corr_data)  - Metric: ' + str(mean), loc = 'center' ,fontsize = 70)
    #plt.tight_layout()
    plt.title( mean )
    
    ax.set_xticks(np.arange(len(var_list)-1))
    ax.set_yticks(np.arange(len(var_list)-1))
    
    ax.set_xticklabels(var_list[1:],fontsize = 45 ,rotation=90)
    ax.set_yticklabels(var_list[1:],fontsize = 45 ,rotation=0)


    plt.savefig(path + '/correlation_matrix_corrected_endcap.png')

    

    plt.close()
    fig, ax = plt.subplots(figsize=(33,33))
    ax.matshow( 100*( data_corr - mc_corr ), cmap = 'bwr', vmin=-5, vmax=5)
    #sns.heatmap(data_corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    #plt.title(f'Correlation Matrix for {key}')    
    
    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate(100*( data_corr - mc_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 60)  
    mean = mean/count   

    ax.set_xticks(np.arange(len(var_list)-1))
    ax.set_yticks(np.arange(len(var_list)-1))
    
    ax.set_xticklabels(var_list[1:],fontsize = 45,rotation=90)
    ax.set_yticklabels(var_list[1:],fontsize = 45,rotation=0)

    ax.set_xlabel(r'(Corr_MC-Corr_data  - Metric: ' + str(mean), loc = 'center' ,fontsize = 70)

    plt.savefig(path + '/correlation_matrix_endcap.png')


#The events are binned in bins of equal number of events of each profilling variable, than the median is calculated!
def plot_profile_barrel( nl_mva_ID, mc_mva_id ,mc_conditions,  data_mva_id, data_conditions, mc_weights, data_weights,path):

    # Barrel only mask!
    mask_mc   = np.abs( mc_conditions[:,1])   < 1.442
    mask_data = np.abs( data_conditions[:,1]) < 1.442 

    nl_mva_ID = nl_mva_ID[mask_mc]
    mc_mva_id = mc_mva_id[mask_mc]
    mc_weights = mc_weights[mask_mc]
    mc_conditions = mc_conditions[mask_mc]

    data_mva_id     = data_mva_id[mask_data]
    data_conditions = data_conditions[mask_data]
    data_weights    = data_weights[mask_data]

    #lets call the function ...
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,0],data_mva_id,data_conditions[:,0],mc_weights,data_weights , path, var = 'pt' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,1],data_mva_id,data_conditions[:,1],mc_weights,data_weights , path, var = 'eta' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,2],data_mva_id,data_conditions[:,2],mc_weights,data_weights , path ,var = 'phi' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,3],data_mva_id,data_conditions[:,3],mc_weights,data_weights , path ,var = 'rho' )



#The events are binned in bins of equal number of events of each profilling variable, than the median is calculated!
def plot_profile_endcap( nl_mva_ID, mc_mva_id ,mc_conditions,  data_mva_id, data_conditions, mc_weights, data_weights,path):

    # Barrel only mask!
    mask_mc   = np.abs( mc_conditions[:,1])   > 1.56
    mask_data = np.abs( data_conditions[:,1]) > 1.56 

    nl_mva_ID = nl_mva_ID[mask_mc]
    mc_mva_id = mc_mva_id[mask_mc]
    mc_weights = mc_weights[mask_mc]
    mc_conditions = mc_conditions[mask_mc]

    data_mva_id     = data_mva_id[mask_data]
    data_conditions = data_conditions[mask_data]
    data_weights    = data_weights[mask_data]

    #lets call the function ...
    plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,0],data_mva_id,data_conditions[:,0],mc_weights,data_weights , path, var = 'pt' )
    #plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,1],data_mva_id,data_conditions[:,1],mc_weights,data_weights , path, var = 'eta' )
    plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,2],data_mva_id,data_conditions[:,2],mc_weights,data_weights , path ,var = 'phi' )
    plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,3],data_mva_id,data_conditions[:,3],mc_weights,data_weights , path ,var = 'rho' )



# Lets do this separatly, first, we do the plots at the barrel only!
def plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,var_mc,data_mva_id,var_data,mc_weights,data_weights,path,var = 'pt' ):
    
    if 'pt' in var:
        bins = np.linspace( 25.0, 65.0, 14 )
    elif 'phi' in var:
        bins = np.linspace( -3.1415, 3.1415, 14)
    elif 'eta' in var:
        bins = np.linspace( -1.442, 1.442, 14 )
    elif 'rho' in var:
        bins = np.linspace( 10.0, 45.0, 14 )

    #arrays to store the 
    position, nl_mean, data_mean, mc_mean = [],[],[],[]
    nl_mean_q25, data_mean_q25, mc_mean_q25 = [],[],[]
    nl_mean_q75, data_mean_q75, mc_mean_q75 = [],[],[]

    for i in range( 0, int(len( bins ) -1) ):

        mva_nl_window     = nl_mva_ID[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mva_mc_window     = mc_mva_id[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mc_weights_window = mc_weights[  ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])  ]

        mva_data_window     = data_mva_id[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 
        data_weights_window = data_weights[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 

        position.append(  bins[i] + (bins[i+1] -bins[i] )/2.   )
        nl_mean.append(   weighted_quantiles_interpolate( mva_nl_window      , mc_weights_window )   )   #np.median(  mva_nl_window )   )
        mc_mean.append(   weighted_quantiles_interpolate( mva_mc_window      , mc_weights_window )   )
        data_mean.append( weighted_quantiles_interpolate( mva_data_window    , data_weights_window ) )

        nl_mean_q25.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles= 0.25  ) )   #np.median(  mva_nl_window )   )
        mc_mean_q25.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles= 0.25  ) )
        data_mean_q25.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles= 0.25  ) )

        nl_mean_q75.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles = 0.75 ) )   #np.median(  mva_nl_window )   )
        mc_mean_q75.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles = 0.75 ) )
        data_mean_q75.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles = 0.75 ) )

    # Plotting the 3 quantiles
    plt.figure(figsize=(10, 6))
    plt.plot( position , nl_mean ,  linewidth  = 2 , color = 'red' , label = 'MC corrected' )
    plt.plot( position , mc_mean ,  linewidth  = 2 , color = 'blue'  , label = 'MC nominal'   )
    plt.plot( position , data_mean , linewidth = 2 , color = 'green', label = 'Data'  )


    plt.plot( position , nl_mean_q25 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q25 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q25 , linewidth = 2 , linestyle='dashed', color = 'green' )

    plt.plot( position , nl_mean_q75 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q75 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q75 , linewidth = 2 , linestyle='dashed', color = 'green' )


    if 'eta' in var:
        plt.xlabel( r'$\eta$' , fontsize = 25 )
    if 'phi' in var:
        plt.xlabel( r'$\phi$' ,  fontsize = 25)
    if 'rho' in var:
        plt.xlabel( r'$\rho$' ,  fontsize = 25 )
    if 'pt' in var:
        plt.xlabel( r'$p_{T}$ [GeV]', fontsize = 25 )

    plt.ylabel( 'Photon MVA ID' )
    plt.legend(fontsize=15)

    plt.ylim( 0.0 , 1.0 )

    if 'eta' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'phi' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'rho' in var:
        plt.ylim( 0.0 , 1.0 )

    plt.tight_layout()

    plt.savefig( path + '/profile_' + str(var) +'.png' )

    plt.close()


def plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,var_mc,data_mva_id,var_data,mc_weights,data_weights,path,var = 'pt' ):
    
    if 'pt' in var:
        bins = np.linspace( 25.0, 80.0, 20 )
    elif 'phi' in var:
        bins = np.linspace( -3.1415, 3.1415, 20)
    elif 'eta' in var:
        bins = np.linspace( [[-2.5,-1.442],  [1.442, 2.5] ], 20 )
    elif 'rho' in var:
        bins = np.linspace( 5.0, 50.0, 20 )

    #arrays to store the 
    position, nl_mean, data_mean, mc_mean   = [],[],[],[]
    nl_mean_q25, data_mean_q25, mc_mean_q25 = [],[],[]
    nl_mean_q75, data_mean_q75, mc_mean_q75 = [],[],[]

    for i in range( 0, int(len( bins ) -1) ):

        mva_nl_window     = nl_mva_ID[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mva_mc_window     = mc_mva_id[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mc_weights_window = mc_weights[  ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])  ]

        mva_data_window     = data_mva_id[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 
        data_weights_window = data_weights[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 

        position.append(  bins[i] + (bins[i+1] -bins[i] )/2.   )
        nl_mean.append(   weighted_quantiles_interpolate( mva_nl_window      , mc_weights_window )   )   #np.median(  mva_nl_window )   )
        mc_mean.append(   weighted_quantiles_interpolate( mva_mc_window      , mc_weights_window )   )
        data_mean.append( weighted_quantiles_interpolate( mva_data_window    , data_weights_window ) )

        nl_mean_q25.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles= 0.25  ) )   #np.median(  mva_nl_window )   )
        mc_mean_q25.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles= 0.25  ) )
        data_mean_q25.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles= 0.25  ) )

        nl_mean_q75.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles = 0.75 ) )   #np.median(  mva_nl_window )   )
        mc_mean_q75.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles = 0.75 ) )
        data_mean_q75.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles = 0.75 ) )

    # Plotting the 3 quantiles
    plt.figure(figsize=(10, 6))
    plt.plot( position , nl_mean ,  linewidth  = 2 , color = 'red' , label = 'MC corrected' )
    plt.plot( position , mc_mean ,  linewidth  = 2 , color = 'blue'  , label = 'MC nominal'   )
    plt.plot( position , data_mean , linewidth = 2 , color = 'green', label = 'Data'  )


    plt.plot( position , nl_mean_q25 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q25 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q25 , linewidth = 2 , linestyle='dashed', color = 'green' )

    plt.plot( position , nl_mean_q75 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q75 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q75 , linewidth = 2 , linestyle='dashed', color = 'green' )


    if 'eta' in var:
        plt.xlabel( r'$\eta$' , fontsize = 25 )
    if 'phi' in var:
        plt.xlabel( r'$\phi$' ,  fontsize = 25)
    if 'rho' in var:
        plt.xlabel( r'$\rho$' ,  fontsize = 25 )
    if 'pt' in var:
        plt.xlabel( r'$p_{T}$ [GeV]', fontsize = 25 )

    plt.ylabel( 'Photon MVA ID' )
    plt.legend(fontsize=15)

    plt.ylim( 0.0 , 1.0 )

    if 'eta' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'phi' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'rho' in var:
        plt.ylim( 0.0 , 1.0 )

    plt.tight_layout()

    plt.savefig( path + '/profile_endcap_' + str(var) +'.png' )

    plt.close()

# This is a plot distribution suitable for dataframes, if one wants to plot tensors see "plot_distributions_for_tensors()"
def plot_distributions__( path, data_df, mc_df, data_weights, mc_weights, variables_to_plot, weights_befores_rw = False ):

    for set in variables_to_plot:

        for key in set:

            mean = np.mean( np.nan_to_num(np.array(data_df[key.replace('_raw','')])) )
            std  = np.std(  np.nan_to_num(np.array(data_df[key.replace('_raw','')])) )

            if( 'Iso' in key or 'DR' in key   ):
                data_hist            = hist.Hist(hist.axis.Regular(100, 0.0, 3.5))
                mc_hist              = hist.Hist(hist.axis.Regular(100, 0.0, 3.5))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, 0.0, 3.5))
            elif( 'hoe' in  key ):
                data_hist            = hist.Hist(hist.axis.Regular(100, 0.0, 0.08))
                mc_hist              = hist.Hist(hist.axis.Regular(100, 0.0, 0.08))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, 0.0, 0.08))
            else:
                data_hist            = hist.Hist(hist.axis.Regular(100, mean - 3.0*std, mean + 4.0*std))
                mc_hist              = hist.Hist(hist.axis.Regular(100, mean - 3.0*std, mean + 4.0*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, mean - 3.0*std, mean + 4.0*std))

            data_hist.fill( np.array(data_df[key.replace('_raw','')]   ) , weight = data_weights )
            
            if( len(weights_befores_rw)  ):
                mc_hist.fill( np.array(  mc_df[key]), weight = weights_befores_rw )
            else:
                mc_hist.fill( np.array(  mc_df[key]) )

            #print( np.shape( np.array(drell_yan_df[key]) ) , np.shape( mc_weights  ) )

            mc_rw_hist.fill( np.array(mc_df[key]) , weight = mc_weights )

            plott( data_hist , mc_hist, mc_rw_hist , path +  str(key) +".png", xlabel = str(key)  )
            
