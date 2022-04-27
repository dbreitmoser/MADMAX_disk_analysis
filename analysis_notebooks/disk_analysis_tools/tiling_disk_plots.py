import disk_analysis_tools.tiling_disk_utils as tdu
import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import seaborn as sns
from pathlib import Path
hexgrid_path = Path('.')/'disk_analysis_tools'/'Klebeplatte_75mm_with_nuten.png' 

def plot_hexgrid(hexgrid_path, ax):
    """Plots Hexagonal map of the gluing machine"""
    im = plt.imread(hexgrid_path)
    #? Hexgrid Dimensions in mm 
    extend_left = -166.13
    extend_right = 166.13
    extend_bottom = -153.23
    extend_top = 153.23
    ax.imshow(im, extent=[extend_left, extend_right, extend_bottom, extend_top])# show grid

def coordinates_plot(dataframe, hexgrid_path=hexgrid_path):
    """Plot only to show the coordinates of given measurement"""
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,12))
    ax.scatter(dataframe['x'], dataframe['y'], s=5, marker='x')
    plot_hexgrid(hexgrid_path)
    ax.set_xlabel("x [mm]", fontsize=20)
    ax.set_ylabel("y [mm]", fontsize=20)
    ax.tick_params(labelsize=20)
    return fig, ax

#? evolved into plot_table_hexagon_flatness
def full_hexagon_plot(
    dataframe,
    hexgrid=True,
    hexagon_origin=False,
    mode='z_mean',
    size=5**2,
    marker='o',
    title='title', 
    cbar_norm = (-100,0,100),
    cmap = 'turbo',
    hexgrid_path = hexgrid_path, 
    figsize=(12,12),
    cbar_title = "height [$\mu$m]",
    fig = None, 
    ax = None,
    gs=None,
    xylabels=True,
    ):
    """Plot the data with colorscale. Choose hexgrid to include the map of the gluing machine"""
    if (fig == None) & (ax == None):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        
    if hexagon_origin: 
        #* plots the central coordinate of each hexagon
        hexgrid_position_path = Path.p.parent / 'coordinates' / 'hexagon-position.txt' 
        origin_df = tdu.read_coords_txt(hexgrid_position_path)
        ax.scatter(origin_df['x'], origin_df['y'], c='tab:green', marker='x') #! hardcoded origin_df!
    
    if hexgrid:
        #* plots the hexagonal map underneath the measurements 
        plot_hexgrid(hexgrid_path, ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    #* Colorscale centered at 0, spans 200 um across
    colornorm = TwoSlopeNorm(vcenter=cbar_norm[1], vmax=cbar_norm[2], vmin=cbar_norm[0]) 

    cp = ax.scatter(dataframe['x'], dataframe['y'], c=dataframe[mode], cmap=cmap, norm=colornorm, s=size, marker=marker)
    c_bar = fig.colorbar(cp, cax=cax) # Add a colorbar to a plot
    c_bar.ax.tick_params(labelsize=18)
    c_bar.ax.set_ylabel(cbar_title, fontsize=18)

    if xylabels:
        ax.set_xlabel("x [mm]", fontsize=18)
        ax.set_ylabel("y [mm]", fontsize=18)
    else: 
        ax.set_xticks([])
        ax.set_yticks([])
    if title!=None:
        ax.set_title(title)
    ax.tick_params(labelsize=18)
    if fig == None:
        return ax
    else: return fig, ax

def plot_table_hexagon_flatness(
    dataframe,
    mode='z_mean',
    size=5**2,
    marker='o',
    title='title', 
    cbar_norm = (-100,0,100),
    offset=10, 
    cmap = 'turbo',
    figsize=(12,12),
    triplet=False,
    fontsize_planarity=14, 
    unit = 'µm'
    ):
    """gluetable datapoints with addtion of x and y shape projections. Choose hexgrid to include the map of the gluing machine"""
  
    fig = plt.figure(figsize=figsize,  constrained_layout=False)
    #* Create 2x3 Grid
    gs = fig.add_gridspec(nrows=2, ncols=3,
                          height_ratios=[3,0.8],
                          width_ratios=[0.8, 3, 0.1],
                          wspace=0.,
                          hspace=0.,
                          right=0.9,
                          top=0.822, 
                          )
    #* populate grid with axes objects
    ax_trip = fig.add_subplot(gs[0, 1])
    ax_xscatt = fig.add_subplot(gs[1, 1], sharex=ax_trip)
    ax_yscatt = fig.add_subplot(gs[0, 0], sharey=ax_trip)
    ax_text = fig.add_subplot(gs[1, 0],)
    #* text axis options
    anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                 va='center', ha='center', fontsize=fontsize_planarity)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_text.spines['bottom'].set_visible(False)
    ax_text.spines['left'].set_visible(False)
    
    std = dataframe[mode].std()
    min_max = dataframe[mode].max() - dataframe[mode].min()
    ax_text.annotate(f'Planarity\nRMS: {LatexFormat(std)} {unit}\nMin-Max: {LatexFormat(min_max)} {unit}', **anno_opts)
    
    #* plot table background picture
    plot_hexgrid(hexgrid_path, ax_trip)
    
    #* initiate color axis (maybe move to its own axis)
    cax1 = plt.subplot(gs[2])
    colornorm = TwoSlopeNorm(vcenter=cbar_norm[1], vmax=cbar_norm[2], vmin=cbar_norm[0])  #* Colorscale centered at 0, spans 200 um across
    
    #* Plot points on table x-y representation
    cp = ax_trip.scatter(dataframe['x'], dataframe['y'], c=dataframe[mode], cmap=cmap, norm=colornorm, s=size, marker=marker)
    c_bar = fig.colorbar(cp, cax=cax1, pad=0.2) # Add a colorbar to a plot
    if title!=None:
        ax_trip.set_title(title)
    #* Confine plot to visible triplet
    if triplet == True:
        triplet_midpoint = [dataframe.x.mean(),dataframe.y.mean()]
        ax_trip.set_xlim(dataframe.x.min() -5, dataframe.x.max() + 5)
        ax_trip.set_ylim(dataframe.y.min()- 5, dataframe.y.max() + 5)
    ax_trip.set_yticklabels([])
    ax_trip.set_xticklabels([])
    c_bar.ax.tick_params(labelsize=18)
    c_bar.ax.set_ylabel(f"height [{unit}]", fontsize=18)
    ax_trip.tick_params(axis="both",direction="in")
    

    ax_xscatt.scatter(dataframe.x, dataframe[mode], s=3 )
    ax_xscatt.tick_params(axis="x", left=True, right=True, labelleft=False, labelright=True)
    ax_xscatt.yaxis.set_label_position("right")
    ax_xscatt.yaxis.tick_right()
    ax_xscatt.grid(True)
    ax_yscatt.scatter(dataframe[mode], dataframe.y, s=3)
    ax_yscatt.tick_params(axis="x", bottom=True, top=True, labelbottom=False, labeltop=True,labelrotation=30)
    ax_yscatt.xaxis.set_label_position("top")
    ax_yscatt.grid(True)
    
    ax_xscatt.tick_params(axis="both",direction="in")
    ax_yscatt.tick_params(axis="both", direction="in")
    ax_xscatt.set_xlabel("x [mm]", fontsize=18)
    ax_yscatt.set_ylabel("y [mm]", fontsize=18)

    # offset = 60
    ax_xscatt.set_ylim(cbar_norm[0] - offset, cbar_norm[2]+offset) 
    ax_xscatt.axhline(0, c='black', ls='--')
    ax_yscatt.set_xlim(cbar_norm[0]-offset,cbar_norm[2]+offset)
    ax_yscatt.axvline(0, c='black', ls='--')
    # plt.tight_layout()

    return fig, (ax_trip, ax_xscatt, ax_yscatt)


#? also not used anymore
def fit_surface_plot(ax,x,y,z,cbar_scale=(-50, 0, 50), show_data=False, data=None):
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    colornorm = TwoSlopeNorm(vcenter=cbar_scale[1], vmax=cbar_scale[2], vmin=cbar_scale[0]) 
    surface = ax.plot_trisurf(x, y, z*1e3, cmap="turbo", norm=colornorm, antialiased=True) #> z in [µm]
    # print("new´plot works")
    if show_data:
        X = data['x']
        Y = data['y']
        Z = data['z_mean'] *1e3 #> [um]
        points = ax.scatter(X, Y, Z, cmap="turbo", norm=colornorm, c=Z)
        
    #c_bar = fig.colorbar(surface, pad=0.15)
    #c_bar.ax.tick_params(labelsize=16)
    #c_bar.ax.set_ylabel("height [$\mu$m]", fontsize=16)
    ax.set_xlabel("x [mm]", fontsize=16)
    ax.set_ylabel("y [mm]", fontsize=16)
    ax.set_zlabel("z [$\mu$m]", fontsize=16)
    ax.tick_params(labelsize=16)

    #return fig, ax 

# def hist_error_band(ax, mean, std, label_std=None, label_mean=None, c_mean=None, c_std=None): 
#     ax.axvspan(mean-std, mean+std, color=c_std, alpha=0.3, label=label_std)
#     ax.axvline(mean, ls='--',c=c_mean, linewidth=2, label=label_mean)
#     return ax

#* UNDER CONSTRUCTION

def LatexFormat(f, scirange=[0.01,1000]):
    # Stolen from Jack Rolph MVP
    if np.isnan(f): return f
    if(np.abs(f)>scirange[0] and np.abs(f)<scirange[1]):
        float_str = f"{f:3.3f}"
    else:
        float_str = f"{f:3.3E}" 
        (base, exponent) = float_str.split("E")
        float_str = r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    return float_str

def hist_label_data(variable):
    import scipy.stats as stats
    from scipy.stats import median_abs_deviation as mad  
    hist_data = {
        "N": len(variable),
        "$\\bar{z}$": np.mean(variable),
        "RMS": np.std(variable),
        # "$\\sigma_{z}$ / $\\sqrt{N}$": np.std(variable) / np.sqrt(len(variable)),
        # "$s_{z}$": stats.skew(variable),
        # "$k_{z}$": stats.kurtosis(variable),
        # "median($z$)":np.median(variable),
        # "MAD($z$)":mad(variable),
        "max-min($z$)": np.max(variable) - np.min(variable),
    }
    
    label = '\n'.join(f'{key} : {LatexFormat(val)}' for key, val in hist_data.items())
    return hist_data, label


#=====================================
#========= Plot time series ==========
#=====================================

def plot_data_vs_time(data, mode="z", figsize=(9,4), n_labels=12, ylabel='z [µm]'):
    import matplotlib.dates as dates
    import pytz
    local_tz = pytz.timezone('Europe/Berlin')

    fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=1)
    sns.lineplot(x='datetime', y=mode, data=data, ci='sd')
    plt.grid(c="grey", ls="-", lw=1, alpha=0.3)

    formatter = dates.DateFormatter('%H:%M', tz=local_tz) 
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.xlabel('time')
    plt.ylabel(ylabel)
    return fig, ax


# def ts_hist(
#     data,
#     mode='z',
#     figsize=(7,6),
#     color_plot='tab:blue', 
#     color_gauss='tab:blue',  
#     fit=False,
#     fit_window=(-100,100),
#     show_datapoints=True,
#     fit_start_param = [ 10, 0, 10],
#     x_label = 'z [µm]', 
#     plot_bins='auto'
#             ): 
#     bins = np.array(range(fit_window[0],fit_window[1], 6)) - 0.5
#     fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
#     _, label = hist_label_data(data[mode])
#     sns.histplot(data[mode], kde=False, ax=ax, element='step', color=color_plot,
#                  stat='probability', label=label, fill=True, alpha=0.4, 
#                  bins=plot_bins,)
    
#     counts,bin_edges = np.histogram(data[mode],bins)
#     counts = counts/counts.sum()
#     bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
#     if show_datapoints:
#         ax.scatter(bin_centres, counts, s=4**2, c='black')
#     if fit:
#         from scipy.optimize import curve_fit
#         def fit_function(x, B, mu, sigma):
#             return ( B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
#         popt, pcov = curve_fit(fit_function, xdata=bin_centres, ydata=counts, p0=fit_start_param)
#         xspace = np.linspace(fit_window[0], fit_window[1], 100000)
#         plt.plot(xspace, fit_function(xspace, *popt),
#             color='tab:orange',
#             ls='--',
#             linewidth=2.5,
#             label=f'$\mu$: {LatexFormat(popt[1])} \n $\sigma$: {LatexFormat(popt[2])}')


#     ax.set_xlabel(x_label)
#     plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.4,  fontsize=14)
#     plt.grid(c="grey", ls="-", lw=1, alpha=0.3)
#     return fig, ax

def ts_hist(
    data,
    mode='z',
    figsize=(7,6),
    color_plot='tab:blue',  
    x_label = 'z [µm]', 
    plot_bins='auto',
    fig_ax = None, 
    hist_stat='count',
    kde=False
            ): 
    if fig_ax == None:
        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    else: fig, ax = fig_ax
    _, label = hist_label_data(data[mode])
    sns.histplot(data[mode], kde=kde, ax=ax, element='step', color=color_plot,
                 stat='count', label=label, fill=True, alpha=0.4, 
                 bins=plot_bins,)

    ax.set_xlabel(x_label)
    plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.4,  fontsize=14)
    plt.grid(c="grey", ls="-", lw=1, alpha=0.3)
    return fig, ax

#? no fits needed anymore ¯\_(ツ)_/¯
# def ts_fit_hist(dataframe,
#         mode='z',
#         bin_plot_range=[-100,100],
#         fit_start_param = [0,40],
#         x_label = 'z [µm]', 
#         fig_title = 'Hist-Fit to data',
#         figsize=(8,8),
#         bin_size = 6, #? µm <- resolution of laser):

#         ): 
#     from iminuit import Minuit
#     from probfit import UnbinnedLH, gaussian, Extended, mid, BinnedLH
#     from matplotlib import gridspec
    
#     data = dataframe[mode].to_numpy()[~(np.isnan(dataframe[mode]))]
#     egauss = Extended(gaussian)
#     # cost functuion
#     unbinned_likelihood = UnbinnedLH(egauss, data, extended=True)
#     # define minimizer function
#     minuit = Minuit(unbinned_likelihood, mean=fit_start_param[0], sigma=fit_start_param[1], N=1000)
#     # minimize function
#     minuit.migrad()
#     # get data from "draw"-function to plot it myself
#     ((data_edges, datay), (errorp, errorm), (total_pdf_x, total_pdf_y), parts) = unbinned_likelihood.draw(minuit, parts=True)
#     plt.clf()
#     # plot fit and data
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], wspace=0.0, hspace=0.0 )
#     ax0 = plt.subplot(gs[0])    # Data points & histogram & fit
#     ax0.tick_params(axis="both", direction="in")
#     nbins = np.arange(bin_plot_range[0],bin_plot_range[1], bin_size)

#     ax0.errorbar(mid(data_edges), datay, errorp, fmt='.', capsize=0, color='black', label='Data')
#     ax0.plot(total_pdf_x, total_pdf_y, color='tab:orange', ls='--', lw=2, label='Unbinned Likelihood')
#     _, label = hist_label_data(dataframe[mode])
#     print(label)
#     sns.histplot(dataframe[mode], kde=False, ax=ax0, element='step', color='tab:blue',
#                     stat='count', label=label, fill=True, alpha=0.4, 
#                     bins=nbins,)
#     ax0.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.4,  fontsize=14)
#     # Residual plot
#     plt.title(fig_title)
#     ax0.grid(c="grey", ls="-", lw=1, alpha=0.3)
#     ax1 = plt.subplot(gs[1], sharex=ax0)
#     unbinned_likelihood.draw_residual(minuit,ax=ax1)
#     # Prettify the plot
#     plt.grid(c="grey", ls="-", lw=1, alpha=0.3)
#     ax1.tick_params(axis="both", direction="in")
#     plt.xlabel(x_label)
#     # show fit parameters
#     boxtext = f"""$\mu$: {LatexFormat(minuit.values[0])}$\pm$  {LatexFormat(minuit.errors[0])}
#     \n$\sigma$: {LatexFormat(minuit.values[1])} $\pm$ {LatexFormat(minuit.errors[1])} """
#     ax0.text(0.15, 0.82, boxtext, transform=fig.transFigure,
#             fontsize=14, verticalalignment="top", 
#             bbox=dict(facecolor="none", edgecolor='none'))
#     return fig, (ax0, ax1)

def plot_triplet_dist_joyplot(all_triplet_data_dict,
                              mode='z_mean',
                              title = "z-Distribution over time",
                              summary_statistics=False,
                              binwidth=6, 
                              time_format = '%H:%M - %d.%m.%Y'): 

    from matplotlib import cm
    n_hours = len(all_triplet_data_dict)
    colors = cm.viridis_r(np.linspace(0, 1, n_hours))
    fig, axes = plt.subplots(n_hours,1, figsize=(6,14))
    i = 0
    z_global_max = 0 
    z_global_min = 0
    for key, data in all_triplet_data_dict.items(): 
        z_min = data.z_mean.min()
        z_max = data.z_mean.max()
        if z_max > z_global_max: z_global_max = np.rint(z_max)
        if z_min < z_global_min: z_global_min = np.rint(z_min)
    # print(z_global_max)
    # print(z_global_min)
    if z_global_max > 500: z_global_max = 100
    if z_global_min < -500: z_global_min = -100
    # print(z_global_max)
    # print(z_global_min)
    
    for (key, data), ax, color in zip(all_triplet_data_dict.items(), axes, colors): 
        # data = data.loc[data.trip_color==triplet_color,:]
        run_nr = int(data.run_nr.unique()[0])
        z_min = data.z_mean.min()
        z_max = data.z_mean.max()
        R = z_max - z_min
        R = np.round(R,2) 
        sig = np.round(data.z_mean.std(),2)
        
        nbins = np.arange(z_global_min, z_global_max ,binwidth)
        sns.histplot(data[mode], kde=False, ax=ax, element='step', color=color,
                    stat='probability', label=f'Hour {run_nr-1}', fill=True, alpha=0.4, 
                    bins=nbins,)
        ax.spines.top.set_color('none')
        ax.spines.right.set_color('none')
        ax.axvline(0,0,1, ls='--', color='black')
        # ax.grid(True)
        plt.subplots_adjust(hspace=-0.3)
        # make background transparent
        rect = ax.patch
        rect.set_alpha(0)
        # remove borders, axis ticks, and labels
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_ylabel('')
        if i == n_hours - 1:
            ax.set_xlabel('z [µm]')
        else:
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_xlabel('')
        i+=1

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax.spines[s].set_visible(False)
        time_str = tdu.calc_measurement_date_and_time(data, time_format=time_format)
        ax.text(z_global_min -2 ,
                0.02,
                f'{time_str}',
                fontweight="bold",
                fontsize=14,
                ha="right",
                va='center')
        if summary_statistics: 
            ax.text(z_global_max+2 ,
                    0.02,
                    f'R:{R}, $\sigma$:{sig}',
                    fontweight="bold",
                    fontsize=14,
                    ha="left",
                    va='center')
    fig.text(0.06,0.867,f"Time",fontsize=14, fontweight="bold",)
    fig.text(0.06,0.89, title ,fontsize=18, fontweight="bold",)
    return fig, axes



def plot_R_RMS_vs_time(result_df): 
    '''takes flatness result DataFrame from calc_flats_statistic_df function
       to display flatness statistics vs time'''
    #TODO: define Error for RMS and turn RMS plot into errorbarplot
    import matplotlib.dates as dates
    import pytz
    local_tz = pytz.timezone('Europe/Berlin')
    fig, axes = plt.subplots(2,1, figsize=(7,7))
    formatter = dates.DateFormatter('%H:%M', tz=local_tz)

    #* Plot min-max (R) vs time
    ax_R = axes[0]
    ax_R.errorbar(x=result_df.datetime, y=result_df.R, yerr=result_df.deltaR,
                fmt='.', capsize=5,)
    ax_R.xaxis.set_major_formatter(formatter)
    ax_R.set_title('R vs time')
    ax_R.set_ylabel('R: Min - Max [µm]')
    ax_R.grid(True)

    #* Plot RMS (std(z_mean)) vs time
    ax_RMS = axes[1]
    ax_RMS.scatter(x=result_df.datetime, y=result_df.RMS, marker='x')
    ax_RMS.xaxis.set_major_formatter(formatter)
    ax_RMS.set_title('RMS vs time')
    ax_RMS.set_ylabel('RMS [µm]')
    ax_RMS.grid(True)
    plt.tight_layout()    
    return fig, axes