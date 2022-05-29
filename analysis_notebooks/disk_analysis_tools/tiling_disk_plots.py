import disk_analysis_tools.tiling_disk_utils as tdu
import numpy as np 
import pandas as pd
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
    plot_hexgrid(hexgrid_path, ax)
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
    cbar_title = "height",
    fig = None, 
    ax = None,
    gs=None,
    xylabels=True,
    unit='µm'
    ):
    """Plot the data with colorscale. Choose hexgrid to include the map of the gluing machine"""
    if (fig == None) & (ax == None):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        
    if hexagon_origin: 
        #* plots the central coordinate of each hexagon
        hexgrid_position_path = Path.p.parent / 'coordinates' / 'hexagon-position.txt' 
        origin_df = tdu.read_coords_txt(hexgrid_position_path)
        ax.scatter(origin_df['x'], origin_df['y'], c='tab:green', marker='x')
    
    if hexgrid:
        #* plots the hexagonal map underneath the measurements 
        plot_hexgrid(hexgrid_path, ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    #* Colorscale centered at 0, spans 200 um across
    colornorm = TwoSlopeNorm(vcenter=cbar_norm[1], vmax=cbar_norm[2], vmin=cbar_norm[0]) 

    cp = ax.scatter(dataframe['x'],
                    dataframe['y'],
                    c=dataframe[mode],
                    cmap=cmap,
                    norm=colornorm,
                    s=size,
                    marker=marker)
    c_bar = fig.colorbar(cp, cax=cax) # Add a colorbar to a plot
    c_bar.ax.tick_params(labelsize=18)
    c_bar.ax.set_ylabel(f'{cbar_title} [{unit}]', fontsize=18)

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
    unit = 'µm', 
    fontsize_ticklabels=16,
    fontsize_title=18,
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
                          top=0.822,)
    
    #* populate grid with axes objects
    ax_trip = fig.add_subplot(gs[0, 1]) #? main plot in the center showing x-y scatter plot of z values in color
    ax_xscatt = fig.add_subplot(gs[1, 1], sharex=ax_trip) #? x projection of main plot located to the bottom of the main plot
    ax_yscatt = fig.add_subplot(gs[0, 0], sharey=ax_trip) #? y projections of main plot located to the left of the main plot
    ax_text = fig.add_subplot(gs[1, 0],) #? plot for planarity annotation 
    #* text axis options
    anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                 va='center', ha='center', fontsize=fontsize_planarity)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_text.spines['bottom'].set_visible(False)
    ax_text.spines['left'].set_visible(False)
    
    std = dataframe[mode].std()
    min_max = dataframe[mode].max() - dataframe[mode].min()
    ax_text.annotate(f'RMS\n{LatexFormat(std)} {unit}\nMin-Max\n{LatexFormat(min_max)} {unit}', **anno_opts)
    
    #* plot table background picture
    plot_hexgrid(hexgrid_path, ax_trip)
    
    #* initiate color axis (maybe move to its own axis)
    cax1 = plt.subplot(gs[2])
    colornorm = TwoSlopeNorm(vcenter=cbar_norm[1], vmax=cbar_norm[2], vmin=cbar_norm[0])  #* Colorscale centered at 0, spans 200 um across
    
    #* Plot points on table x-y representation
    cp = ax_trip.scatter(dataframe['x'], dataframe['y'], c=dataframe[mode], cmap=cmap, norm=colornorm, s=size, marker=marker)
    ax_trip.tick_params(labelsize=fontsize_ticklabels)
    c_bar = fig.colorbar(cp, cax=cax1, pad=0.2) # Add a colorbar to a plot
    ax_trip.tick_params(which='both', labelbottom=False, labelleft=False)
    if title!=None:
        ax_trip.set_title(title, pad=15, fontweight='bold')
    #* Confine plot to visible triplet
    if triplet == True:
        triplet_midpoint = [dataframe.x.mean(),dataframe.y.mean()]
        ax_trip.set_xlim(dataframe.x.min() -5, dataframe.x.max() + 5)
        ax_trip.set_ylim(dataframe.y.min()- 5, dataframe.y.max() + 5)
    # ax_trip.set_yticklabels([])
    # ax_trip.set_xticklabels([])
    c_bar.ax.tick_params(labelsize=fontsize_ticklabels)
    c_bar.ax.set_ylabel(f"height [{unit}]", fontsize=fontsize_title)
    ax_trip.tick_params(axis="both",direction="in")

    ax_xscatt.scatter(dataframe.x, dataframe[mode], s=3)
    ax_xscatt.tick_params(axis="x", left=True, right=True, labelleft=False, labelright=True, labelbottom=True)
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
    # Stolen from Jack Rolph MVP
    import scipy.stats as stats
    from scipy.stats import median_abs_deviation as mad  
    hist_data = {
        "N": len(variable),
        "$\\bar{z}$": np.mean(variable),
        "RMS": np.std(variable),
        "max-min($z$)": np.max(variable) - np.min(variable),
        # "$\\sigma_{z}$ / $\\sqrt{N}$": np.std(variable) / np.sqrt(len(variable)),
        # "$s_{z}$": stats.skew(variable),
        # "$k_{z}$": stats.kurtosis(variable),
        # "median($z$)":np.median(variable),
        # "MAD($z$)":mad(variable),
    }
    
    label = '\n'.join(f'{key} : {LatexFormat(val)}' for key, val in hist_data.items())
    return hist_data, label


#=====================================
#========= Plot time series ==========
#=====================================

def plot_data_vs_time(data, mode="z", figsize=(9,4), n_labels=12, ylabel='z', unit='µm', title='title'):
    import matplotlib.dates as dates
    import pytz
    local_tz = pytz.timezone('Europe/Berlin')

    fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=1)
    sns.lineplot(x='datetime', y=mode, data=data, ci='sd')
    plt.grid(c="grey", ls="-", lw=1, alpha=0.3)
    ax.set_title(title)
    formatter = dates.DateFormatter('%H:%M', tz=local_tz) 
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.xlabel('time')
    plt.ylabel(f'{ylabel} [{unit}]')
    return fig, ax




def ts_hist(
    data,
    mode='z',
    figsize=(7,6),
    color_plot='tab:blue',  
    x_label = 'z', 
    plot_bins='auto',
    fig_ax = None, 
    hist_stat='count',
    kde=False, 
    unit='µm',
    title='title'
            ): 
    if fig_ax == None:
        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    else: fig, ax = fig_ax
    _, label = hist_label_data(data[mode])
    sns.histplot(data[mode], kde=kde, ax=ax, element='step', color=color_plot,
                 stat='count', label=label, fill=True, alpha=0.4, 
                 bins=plot_bins,)

    ax.set_xlabel(f'{x_label} [{unit}]')
    plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.4,  fontsize=14)
    plt.grid(c="grey", ls="-", lw=1, alpha=0.3)
    if title!=None:
        ax.set_title(title)
    return fig, ax

def plot_dist_joyplot(all_triplet_data_dict,
                              triplet_color = 'all',
                              mode='z_mean',
                              title = "z-Distribution over time",
                              timeaxis='time_h',
                              summary_statistics=False,
                              binwidth=6, 
                              figsize=(6,12),
                              time_format = '%H:%M - %d.%m.%Y',
                              ):

    from matplotlib import cm
    stat_result_df = tdu.calc_flats_statistic_df(all_triplet_data_dict)
    plot_runs = stat_result_df.loc[stat_result_df.odd_runs==1]
    n_hours = stat_result_df.odd_runs.value_counts()[1]
    
    colors = cm.viridis_r(np.linspace(0, 1, n_hours))
    fig, axes = plt.subplots(n_hours ,1, figsize=figsize)
    i = 0
    z_global_max = 0 
    z_global_min = 0
    for key, data in all_triplet_data_dict.items(): 
        z_min = data[mode].min()
        z_max = data[mode].max()
        if z_max > z_global_max: z_global_max = np.rint(z_max)
        if z_min < z_global_min: z_global_min = np.rint(z_min)

    
    for row, ax, color in zip(plot_runs.itertuples(), axes, colors): 
        run_nr = row.run_nr
        if triplet_color=='all':
            data = all_triplet_data_dict[f'run_nr_{run_nr}']
        else:
            data = all_triplet_data_dict[f'run_nr_{run_nr}']
            data = data.loc[data.trip_color==triplet_color,:]
        z_min = data[mode].min()
        z_max = data[mode].max()
        R = z_max - z_min
        R = np.round(R,2) 
        sig = np.round(data[mode].std(),2)
        
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
        if timeaxis == 'datetime':
            time_str = tdu.calc_measurement_date_and_time(data, time_format=time_format)
            fig.text(0.06,0.867,f"Time",fontsize=14, fontweight="bold",)
        else:
            time_str = row.time_h
            fig.text(0.06,0.867,f"Time passed [h]",fontsize=14)
        ax.text(z_global_min - 2 ,
                0.02,
                f'{time_str:.2f}',
                fontweight="bold",
                fontsize=14,
                ha="right",
                va='center')
        if summary_statistics: 
            ax.text(z_global_max+2 ,
                    0.02,
                    f'$\sigma$:{sig} µm\nmin-max:{R} µm',
                    # fontweight="bold",
                    fontsize=14,
                    ha="left",
                    va='center')
    fig.text(0.06,0.89, title ,fontsize=18, fontweight="bold",)
    return fig, axes



def plot_R_RMS_vs_time(meas_dict_pt, timeaxis='time_h', R_range=70, RMS_range=10,): 
    '''takes flatness result DataFrame from calc_flats_statistic_df function
       to display flatness statistics vs time'''
       
    result_df = tdu.calc_flats_statistic_df(meas_dict_pt)
    import matplotlib.dates as dates
    import pytz
    local_tz = pytz.timezone('Europe/Berlin')
    fig, axes = plt.subplots(2,1, figsize=(7,7))
    #* Plot min-max (R) vs time
    ax_R = axes[0]
    ax_R.errorbar(x=result_df[timeaxis], y=result_df.R, yerr=result_df.deltaR,
                fmt='.', capsize=5,)
    if timeaxis=='datetime':
      formatter = dates.DateFormatter('%H:%M', tz=local_tz)
      ax_R.xaxis.set_major_formatter(formatter)
    else: ax_R.set_xlabel('time [hours passed]')
    ax_R.set_ylabel('Min-Max [µm]')
    ax_R.grid(True)
    R_mean = result_df.R.mean()
    R_ylim_lower = R_mean - np.floor(R_range/2)
    R_ylim_upper = R_mean + np.floor(R_range/2)
    ax_R.set_ylim(R_ylim_lower,R_ylim_upper)
   #  print(R_mean)
   #  print(R_ylim_lower)
   #  print(R_ylim_upper)
    ax_R.fill_between(result_df[timeaxis], R_mean - 12, R_mean + 12, alpha=0.2, color='tab:orange')
    ax_R.set_title('Min-Max vs time')
  
    #* Plot RMS (std(z_mean)) vs time
    ax_RMS = axes[1]
    ax_RMS.scatter(x=result_df[timeaxis], y=result_df.RMS, marker='x')
    if timeaxis=='datetime':
      ax_RMS.xaxis.set_major_formatter(formatter)
      ax_RMS.set_xlabel('datetime')
    else: ax_RMS.set_xlabel('time [hours passed]')
    
    ax_RMS.set_title('RMS vs time')
    ax_RMS.set_ylabel('RMS [µm]')
    ax_RMS.grid(True)
    RMS_mean = result_df.RMS.mean()
    RMS_ylim_lower = RMS_mean - np.floor(RMS_range/2)
    RMS_ylim_upper = RMS_mean + np.floor(RMS_range/2)
   #  print(f'RMS_lower = {RMS_ylim_lower}')
   #  print(f'RMS_ylim_upper = {RMS_ylim_upper}')
    if RMS_ylim_lower < 0: RMS_ylim_lower = 0
    ax_RMS.set_ylim(RMS_ylim_lower, RMS_ylim_upper)
    ax_RMS.axhline(RMS_mean, ls='--', c='black')
    
    plt.tight_layout()    
    return fig, axes

def lw_repr(width, norm_width=300): return 10* (np.log(width/norm_width)+0.5)
def plot_microscope_glue_canals(dataframe,
                                mode='gap_relative',
                                title='Steelplate gap data',
                                cbar_norm = (-100,0,100),
                                cmap = 'turbo',
                                hexgrid_path = hexgrid_path,
                                figsize=(12,12),
                                cbar_title = "glue penetration depth [$\mu$m]",
                                lw_repr = lw_repr):
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # #* Colorscale centered at 0, spans 200 um across
    colornorm = TwoSlopeNorm(vcenter=cbar_norm[1], vmax=cbar_norm[2], vmin=cbar_norm[0]) 

    cp = ax.scatter(dataframe['x'], dataframe['y'], c=dataframe[mode],
                    cmap=cmap, norm=colornorm, s=1, marker='o', alpha=1)
    c_bar = fig.colorbar(cp, cax=cax) # Add a colorbar to a plot
    c_bar.ax.tick_params(labelsize=18)
    c_bar.ax.set_ylabel(cbar_title, fontsize=18)
    plot_hexgrid(hexgrid_path, ax)
    for row in dataframe.itertuples():   #? Itterate over rows and be able to acces row attributes as usual
        xs = [row.x, row.adj_x1, row.adj_x2]
        ys = [row.y, row.adj_y1, row.adj_y2]

        ax.plot(xs, ys,
                c=cp.to_rgba(dataframe.loc[dataframe.arm_label==row.arm_label,:].gap_relative), #* color from scatterplot
                lw=f'{lw_repr(row.glue_width)}' #* linewidth represents width of glue trace on triplet
                )
    if title!=None:
        ax.set_title(title)

    ax.set_xlabel("x [mm]", fontsize=18)
    ax.set_ylabel("y [mm]", fontsize=18)
    return fig,ax

def control_plots(df, z_col='z', hist_log=True, unit='µm',title='title'):
    plot_data_vs_time(df, mode=z_col, unit=unit, title=title)
    ts_hist(df, mode=z_col, unit=unit, title=title)
    if hist_log:
        plt.yscale('log')

def vac_mapping_helper(vac_str):
    import re
    pattern = re.compile(r'\w*[0-9]+')
    matches = pattern.findall(vac_str)

    if matches[-1] == '1000':
        return 'vac off'
    if matches[-1] == '60':
        return 'vac on'
    else:
        return 'vac {matches[1]} mbar'
    
def title_str_from_metadata(meta_data,meas_id_sig,meas_id_bg,):
    def one_data_title_str_helper(meta_data,meas_id):
        meas_id_info = meta_data.loc[meta_data.measurement_id==meas_id]
        if meas_id_info.material.values[0] == 'table':
            return 'table'
        process_step = meas_id_info.process_step.values[0]
        vac =  meas_id_info.vac_mapping.values[0]
        vac = vac_mapping_helper(vac)
        return f'({process_step}, {vac})'
    sig_string = one_data_title_str_helper(meta_data,meas_id_sig)
    bg_string = one_data_title_str_helper(meta_data,meas_id_bg)
    print(sig_string)
    title_string = '$z_{signal}$ - $z_{ref}$\n'+f'{sig_string} - {bg_string}'
    return title_string

def plot_analysis_results(exp_id:int,
                          meas_dict_pt:dict,
                          exp_db:pd.DataFrame,
                          meta_data:pd.DataFrame,
                          meas_id_sig:int=1,
                          meas_id_bg:int=1,
                          joyplot=True,
                          joyplot_summary_stats=False,
                          R_vs_t=True,
                          hexagon_flatness=True,
                          plot_runs=None, # otherwise list of runs to plot
                          triplet=True): 
    #*=============== hexagon flatness plot ===================
    if plot_runs==None:
        plot_runs=np.arange(1,len(meas_dict_pt.keys())+1,1)
    if triplet:
        example_run = meas_dict_pt['run_nr_1']
        colors = example_run.trip_color.unique()
    if hexagon_flatness:
        title_string = title_str_from_metadata(meta_data, meas_id_sig, meas_id_bg)
        stats_df = tdu.calc_flats_statistic_df(meas_dict_pt)
        for row in stats_df.itertuples(): 
            run_nr = row.run_nr
            if run_nr in plot_runs:
                if triplet:
                    for col in colors:
                        data = meas_dict_pt[f'run_nr_{run_nr}']
                        plot_df = data[data.trip_color==col]
                        fig, ax = plot_table_hexagon_flatness(plot_df,
                                                            mode=('z_mean'),
                                                            size=5**2,
                                                            cbar_norm=(-50, 0, 50),
                                                            triplet=triplet,
                                                            title=title_string,
                                                            figsize=(7,7),
                                                            fontsize_ticklabels=14, 
                                                            fontsize_title=14)
                        exp_description_str_short = exp_db.loc[exp_db.exp_id == exp_id].exp_description_short.values[0]
                        fig.text(0.05,0.901, f'{exp_description_str_short}\nhour {row.time_h:.2f}\n{col} triplet' ,fontsize=14,)
                else:
                    data = meas_dict_pt[f'run_nr_{run_nr}']
                    fig, ax = plot_table_hexagon_flatness(data,
                                                                mode=('z_mean'),
                                                                size=5**2,
                                                                cbar_norm=(-50, 0, 50),
                                                                triplet=triplet,
                                                                title=title_string)
                    exp_description_str_short = exp_db.loc[exp_db.exp_id == exp_id].exp_description_short.values[0]
                    fig.text(0.05,0.865, f'{exp_description_str_short}\nhour {row.time_h:.2f}' ,fontsize=16,)
       #*=============== R vs t plot ===================
    if R_vs_t:
        plot_R_RMS_vs_time(meas_dict_pt)
       #*=============== joyplot ===================
    if joyplot:
        if triplet:
            for col in colors:
                fig_joy, ax_joy = plot_dist_joyplot(meas_dict_pt,
                                                    triplet_color=col,
                                                    timeaxis='time_h,',
                                                    time_format='%H:%M',
                                                    figsize=(5,10),
                                                    summary_statistics=joyplot_summary_stats)
                fig_joy.text(0.78,0.901, f'{col} triplet' ,fontsize=16,)
        else:
            plot_dist_joyplot(meas_dict_pt,
                              triplet_color='all',
                                timeaxis='time_h,',
                                time_format='%H:%M',
                                figsize=(5,10),
                                summary_statistics=joyplot_summary_stats)
            
def plot_z_table_hist(meas_pt_df, meta_data, meas_id_sig=1, meas_id_bg=1, mode='z_mean'): 
    plot_bins = np.arange(meas_pt_df[mode].min(), meas_pt_df[mode].max(),6)
    title = title_str_from_metadata(meta_data,
                                meas_id_sig, #* measurement id signal
                                meas_id_bg, #* measurement id backgroundmeas_id_sig,meas_id_bg
                                )
    fig, ax = ts_hist(meas_pt_df, mode=mode, plot_bins=plot_bins, title=title)
    return fig, ax