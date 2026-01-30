# script made to plot the main validation and resulting distributions

# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
mplhep.style.use([mplhep.style.CMS])
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

def plott_noratio(data_hist,mc_hist,mc_rw_hist ,output_filename,xlabel,text=None):
    plt.close()
    fig, ax = plt.subplots()
    
    if not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object")
        
    hep.histplot(
                mc_hist,
                density=True,
                label = r'Simulation',
                color = "blue",
                linewidth=3,
                ax=ax,
                flow='sum'
            )

    hep.histplot(
                mc_rw_hist,
                density=True,
                label=r'Simulation (Corr)',
                color = "None",
                linewidth=3,
                linestyle='--',
                ax=ax,
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
            ax=ax,
            flow='sum'
        )

    ax.margins(y=0.15)
    ax.set_ylim(0, 1.15*ax.get_ylim()[1])
    ax.tick_params(labelsize=22)

    if( "integral" in str(xlabel)  ):
        xlabel_new = "light integral [counts]"
        ax.set_xlabel( str(xlabel_new) , fontsize=26)
    elif( "tgausssigma" in str(xlabel)  ):
        xlabel_new = r'\sigma_{t} [pix]'
        ax.set_xlabel( str(xlabel_new) , fontsize=26)
    elif( "nhits" in str(xlabel)  ):
        xlabel_new = r'n_{hits}'
        ax.set_xlabel( str(xlabel_new) , fontsize=26)
    else:
        xlabel_new = xlabel.replace("sc_", "")
        ax.set_xlabel( str(xlabel_new) , fontsize=26)
    
    ax.tick_params(labelsize=24)

    # Create a custom legend handle to show a line
    from matplotlib.lines import Line2D
    line = Line2D([0], [0], color='blue', linewidth=3)

    # Get existing legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Replace the handle for the first histogram with the custom line
    handles[1] = line

    # Add the legend with the modified handles
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=20)
    ax.text(0.05, 0.96, r'$^{55}Fe$ data/sim', transform=ax.transAxes, fontsize=20, verticalalignment='top')
    if text:
        ax.text(0.50,0.80, text, transform=ax.transAxes, fontsize=20, verticalalignment='top')

    ax.margins(x=0)

    # Adjust the tight_layout to not add extra padding
    fig.tight_layout(h_pad=0, w_pad=0)

    print(f"===> Validation save plot {output_filename}.pdf/png")
    for ext in ["png","pdf"]:
        fig.savefig(f"{output_filename}.{ext}")

    
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

def plot_distributions( path, variables_to_plot, data_df, mc_df, corr_df=None, params=None, doratio=False, suffix=None ):

    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    for variable in variables_to_plot:

        # Extract data for the variable
        data_values = data_df[variable].dropna().values
        mc_values = mc_df[variable].dropna().values
        if corr_df is not None:
            corr_values = corr_df[variable].dropna().values
        
        # Compute mean and standard deviation for data histogram binning
        mean = np.mean(data_values)
        std = np.std(data_values)

        # Define histogram bin edges based on the variable name
        if 'Iso' in variable or 'DR' in variable:
            bin_edges = np.linspace(0.0, 3.5, 101)
        elif 'hoe' in variable:
            bin_edges = np.linspace(0.0, 0.08, 101)
        else:
            bin_min = max(0,mean - 5.0 * std)
            bin_max = mean + 5.0 * std
            bin_edges = np.linspace(bin_min, bin_max, 51)

        # Create histograms
        bins = hist.axis.Variable(bin_edges)
        data_hist = hist.Hist(bins)
        mc_hist = hist.Hist(bins)
        corr_hist = hist.Hist(bins)

        # Fill histograms
        data_hist.fill(data_values)
        mc_hist.fill(mc_values)
        if corr_df is not None:
            corr_hist.fill(corr_values)
        else:
            corr_hist.fill(mc_values)

        text=None
        if params:
            text = f"z = {params['ztrue_val']} cm\nSim: $\lambda_{{abs}}$={params['lambda_val']}mm, $\\alpha$={params['alpha_val']}\nData: P={params['P_val']}bar, T={params['T_val']}C"
        # Plot and save the histograms
        suff=f"_{suffix}" if suffix else ""
        suff = suff.replace(".","p")
        output_path = os.path.join(path, f"{variable}{suff}")
        if doratio:
            plott(data_hist, mc_hist, corr_hist, output_path, xlabel=variable)
        else:
            plott_noratio(data_hist, mc_hist, corr_hist, output_path, xlabel=variable, text=text)


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


# Converting covariance to correlation matrices
def cov_to_corr(cov_matrix):
    stddev = torch.sqrt(torch.diag(cov_matrix))
    stddev_matrix = torch.diag_embed(stddev)
    corr_matrix = torch.inverse(stddev_matrix) @ cov_matrix @ torch.inverse(stddev_matrix)
    return corr_matrix

