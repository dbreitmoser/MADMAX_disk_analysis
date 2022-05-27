import pandas as pd
import numpy as np 
import pytz
local_tz = pytz.timezone('Europe/Berlin')
from disk_analysis_tools import tiling_disk_plots as tdp

#========================================================
#       Reading in Data
#========================================================
def read_coords_txt(filename):
    df = pd.read_csv(filename, delimiter=' ', decimal=',', header=None, names=['hex_nr','point','x','y'])
    return df 

#! Depreached, newer measurement files contain two more columns: 'run_nr' and 'unix_time'
def read_measurement_txt_old(filename):
    df = pd.read_csv(filename, delimiter=' ', decimal=',', header=None,
                     names=['hex_nr','point','x','y' ,'z'])
    return df

def read_measurement_txt(file_path):
    df = pd.read_csv(file_path, delimiter=' ', decimal=',', header=1,
                     names=['run_nr','hex_nr','point','x','y' ,'z', 'unix_time'])
    return df

def read_measurement_csv(filepath):
    df = pd.read_csv(filepath, delimiter=' ', decimal=',', header=1,
                names=['run_nr','hex_nr','point','x','y' ,'z', 'unix_time'])
    return df

def read_single_measurement(folder, filename, old=False):
    from pathlib import Path
    file_path = Path('.') / folder / filename
    if old:
        df = read_measurement_txt_old(file_path)
        return df
    df = read_measurement_txt(file_path)
    print('Reading data...')
    df = df_convert_unix_to_datetime(df)
    print('Reading done')
    return df


def read_txt_files(folder, old=False):
    from os import walk
    from pathlib import Path
    """reads files in a folder and returns a dict with: 
        data_dict = {filename: data}"""
    file_path = Path('.') / folder
    file_list = next(walk(file_path), (None, None, []))[2]  # [] if no file
    data_dict = {file: read_single_measurement(file_path, file, old) for file in file_list}       
    return data_dict


#========================================================
#       helper functions
#========================================================
def key_sort_helper(string):
    # function to help sort the measurement dict 
    #? Function extractes number of string of type "0mbar" eg:  key_sort_helper("50mbar") >> returns 50
    import re
    match = re.match(r"([0-9]+)([a-z]+)", string, re.I)
    return match.group(1)

def dict_date_to_datetime(date_dict: dict, date_str: str='%H:%M:%S %d-%m-%Y'): 
    import datetime
    date_dict = {key: datetime.datetime.strptime(val, date_str) for key, val in date_dict.items()}
    date_dict = {key: local_tz.localize(val) for key, val in date_dict.items()}
    return date_dict

#========================================================
#       Manipulations on dataframes
#========================================================
def label_ring(row): 
    #? one Hexagon has 4 rings of points. 1 is the most inner one, 4 is the most outer
    ring_1 = [1, 36, 19, 18]
    ring_2 = [2,3,4,33,34,35,20,21,22,15,16,17]
    ring_3 = [9,8,7,6,5, 32,31,30,29,28,
            27,26,25,24,23, 14,13,12,11,10]
    ring_4 = list(range(37, 61, 1))
    
    if row['point'] in ring_1: return 1
    elif row['point'] in ring_2: return 2
    elif row['point'] in ring_3: return 3
    elif row['point'] in ring_4: return 4
    else: return ValueError('Point not in Rings')

def add_ring_nr_label(dataframe): 
    dataframe['ring_nr'] = np.nan
    dataframe['ring_nr'] = dataframe.apply(lambda row: label_ring(row), axis=1)
    return dataframe

def label_trip_color(row): 
    trip_color_dict = {
        'blue': [18, 19, 7],
        'orange': [17,16,6],
        'brown': [15, 14, 5],
        'grey': [4, 13, 12],
        'yellow': [3, 10, 11],
        'green': [8, 9, 2],
    }
    for color_name, color_hexagons in trip_color_dict.items():
        if row['hex_nr'] in color_hexagons: return color_name
    else: return 'white'
        
def add_triplet_color_label(dataframe): 
    dataframe['trip_color'] = np.nan
    dataframe['trip_color'] = dataframe.apply(lambda row: label_trip_color(row), axis=1)
    return dataframe


def filter_data(df,
                point_cut=True, # Cuts point 1
                mm_cut=True, # Cuts at -100mm
                median_hex_cut=True, # Cuts at 100µm from median per hexagon
                ):
    """Applies standard cuts(filters) on the raw z data
       returns: filtered dataframe"""
    #* Apply "point" 1 cut: 
    if point_cut:
      df = df.loc[df.point != 1,:]  
    #* Apply -100mm cut
    #* --- just precautious, usualy everything is caught by cut before
    if mm_cut:
      df = df.loc[df.z < -100]
    #* Remove all data +/- 100µm of median of each hexagon
    if median_hex_cut:
      df = remove_outliers(df, cut_threshold=0.1)#? ct in mm
    return df

def remove_outliers(df, mode='z', cut_threshold=100):
    """df as usual, cut_threshold in µm -> z also in µm!!!!
    returns: df with z_clean -> z within +-100µm of median of each
        hexagon """
    for hexagon in df.hex_nr.unique():
        temp_df = df.loc[df.hex_nr == hexagon,:]
        hex_mean = temp_df[mode].median()
        lower_cut = hex_mean - cut_threshold
        upper_cut = hex_mean + cut_threshold
        # print(f'hex_nr = {hexagon}')
        # print(f'hex_mean = {hex_mean}')
        # print(f'lower_cut = {lower_cut}')
        # print(f'upper_cut = {upper_cut}')
        def temp_sort_func(z):
            return  z if lower_cut < z < upper_cut else np.NaN
        df[mode] = df.loc[:,mode].apply(lambda z: temp_sort_func(z))
    return df 


#========================================================
#       Error Definition
#========================================================
LAO_ERROR = 0.0 #*12*1e-3 #m
def nan_mean(array):
    # return mean even if array has NaN values
    masked_array = np.ma.array(array, mask=np.isnan(array)) # Use a mask to mark the NaNs
    return np.mean(masked_array)

def nan_std(array):
    # return std even if array has NaN values
    masked_array = np.ma.array(array, mask=np.isnan(array)) # Use a mask to mark the NaNs
    return np.std(masked_array)

def mean_error(data): 
    return nan_std(data) / np.sqrt(13) #? 13 measurements in one single point 

def measurement_error(data): 
    return np.sqrt(mean_error(data)**2 + LAO_ERROR**2)
#========================================================

def point_table(dataframe, ring=False, z_col='z'):
    """returns DF pivot table with mean, std and median of z by points on the hexagon"""
    if ring: 
        point_table = pd.pivot_table(dataframe,
                                    values=['x','y',f'{z_col}', 'unix_time', 'ring_nr'],
                                    index=['hex_nr', 'point'],
                                    aggfunc={
                                        'x': np.mean,
                                        'y': np.mean,
                                        f'{z_col}': [nan_mean, measurement_error], #? mesurement Error = sqrt(standard_error + systematic_error)
                                        'unix_time': np.mean, 
                                        'ring_nr': np.max
                                        }
                                    ) 
        point_table[('x','mean')] = point_table[('x','mean')].apply(lambda x: round(x,2))
        point_table[('y','mean')] = point_table[('y','mean')].apply(lambda x: round(x,2))
        point_table.columns = ['_'.join(col).rstrip('_') for col in point_table.columns.values]
        point_table.rename(columns = {'x_mean':'x', 'y_mean':'y',
                                      'ring_nr_amax':'ring_nr',
                                      'unix_time_mean': 'unix_time',
                                      f'{z_col}_nan_mean': 'z_mean',
                                      f'{z_col}_measurement_error': 'z_err'
                                      }, inplace = True)
        point_table.reset_index(inplace=True)
        return point_table
    else: 
        point_table = pd.pivot_table(dataframe,
                            values=['x','y',f'{z_col}', 'unix_time'],
                            index=['hex_nr', 'point'],
                            aggfunc={
                                'x': np.mean,
                                'y': np.mean,
                                f'{z_col}': [nan_mean, measurement_error], #? Here I choose mean and std -> mesurement Error = sqrt(standard_error + systematic_error)
                                'unix_time': np.mean, 
                                }
                            ) 
        point_table[('x','mean')] = point_table[('x','mean')].apply(lambda x: round(x,2))
        point_table[('y','mean')] = point_table[('y','mean')].apply(lambda x: round(x,2))
        point_table.columns = ['_'.join(col).rstrip('_') for col in point_table.columns.values]
        point_table.rename(columns = {'x_mean':'x', 'y_mean':'y',
                                      'ring_nr_amax':'ring_nr',
                                      'unix_time_mean': 'unix_time',
                                      f'{z_col}_nan_mean': 'z_mean',
                                      f'{z_col}_measurement_error': 'z_err'
                                      }, inplace = True)
        point_table.reset_index(inplace=True)
        return point_table


def query_temps_from_db(db_path, starttime, endtime):
    """Reads the temperature and humidity data from the Logger_MADMAX.db
        returns a pandas dataframe with relevant data. 
        Start and end time in the following format: 'YYYY-MM-DD hh:mm:ss'
        Eg: Starttime: '2021-10-18 16:15:49', Endtime: '2021-10-19 07:01:58' """
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    #* Query SQL data
    sql_query = f"""SELECT  *
    from manufact_logger
    where timestamp between '{starttime}' and '{endtime}' """

    c.execute(sql_query)
    t = c.fetchall()
    #* Read into dataframe
    temp_humid_db = pd.DataFrame(t,
                        columns=["date",
                                "T_plate",
                                "T_ambient",
                                "humidity", 
                                "dewpoint", 
                                "fan_speed"]
                            )
    temp_humid_db['datetime'] = temp_humid_db['date'].apply(lambda x: pd.to_datetime(x))
    temp_humid_db['datetime'].dt.tz_localize('Europe/Berlin')
    return temp_humid_db

def query_exp_db(db_path='measurements.db'):
    '''retrieve experiments descriptions from measurements.db'''
    import sqlite3 as sql
    import pandas as pd
    with sql.connect(db_path) as conn: 
        c = conn.cursor()
        
        #* Query SQL data
        sql_query = f"""SELECT  *
        from experiments """

        c.execute(sql_query)
        t = c.fetchall()
        
    #* Read into dataframes
    exp_df = pd.DataFrame(t,
                        columns=["exp_id",
                                "exp_description",
                                "exp_description_short"])
    return exp_df


def fetch_meas_metadata(exp_id=1, db_path='measurements.db'): 
    '''retrieve experiments descriptions from measurements.db'''
    import sqlite3 as sql
    import pandas as pd
    with sql.connect(db_path) as conn: 
        c = conn.cursor()
        
        #* Query SQL data
        sql_query = f"""SELECT  *
        from measurement_meta WHERE exp_id == {exp_id} """

        c.execute(sql_query)
        meas_meta = c.fetchall()

    #* Read into dataframe
    meta_df = pd.DataFrame(meas_meta,
                        columns=['measurement_id',
                                'exp_id',
                                'date',
                                'file_name',
                                'material', 
                                'process_step', 
                                'vac_mapping', 
                                'coordinates',
                                'meas_cap_status',
                                'comments'
                                ])
    return meta_df


def df_convert_unix_to_datetime(df: pd.DataFrame):
    """Takes dataframe with unix timestamp and replaces by python datetime object + converts to berlin timezone.
       new column: 'date' -> datetime.datetime object (UTC +2 europe/berlin) """
       
    df['datetime'] = df['unix_time'].apply(lambda x: pd.to_datetime(x, unit='s', utc=True))
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/Berlin')
    # df.drop('unix_time', axis=1)
    return df

def add_column_time_passed(df: pd.DataFrame):
    import time
    """Add column of passed time in hours; df needs datetime Column !"""
    df['unix_time'] = df['datetime'].apply(lambda x: int(time.mktime(x.timetuple())))
    df['time_h'] = df['unix_time'].apply(lambda x: (x - df['unix_time'].iloc[0]) / (60*60))
    return df

#========================================================
#           Stuff with manipulating measurements
#========================================================

def quad_sum(s_1, s_2):
    '''Sums two values in quadrature -> errorpropagation'''
    s_1_sq = np.power(s_1,2)
    s_2_sq = np.power(s_2,2)
    sums = np.add(s_1_sq, s_2_sq).astype(float)
    return np.sqrt(sums)


# #! Depreached-------------------------------
# def combine_mean_measurements(data_signal, data_background, z_mean_col='z_mean', z_err_col='z_err'):
#     # Takes only point_tables
#     delta_df = data_signal.copy()
#     delta_df[z_mean_col] = data_signal[z_mean_col] - data_background[z_mean_col]

#     if 'unix_time' in data_signal.keys(): 
#     #     delta_df['datetime'] = data_signal['datetime']
#         delta_df['unix_time'] = data_signal['unix_time']
#     delta_df[z_err_col] = quad_sum(data_signal[z_err_col], data_background[z_err_col])
#     return delta_df 
# #!-----------------------------------------
# substitution for old combine_mean_measurements
def subtract_mean_measurements(df_signal,
                               df_background,
                               z_mean_col='z_mean',
                               z_err_col='z_err'): 
    '''input: 
        - df_signal: filtered singal point table of measurement
        - df_background: filtered background point table of measurement
        - z_mean_col: name of z_mean column
        - z_err_col: name of z_error column
        
        returns: delta df
            - merged dataframe on matching hexagon and point number
            - adds column of subtracted mean z values as z_mean_res 
            - adds column of errorpropagated z measurement error'''
    if not ('hex_point' in df_signal.keys()):
        df_signal['hex_point'] = df_signal.apply(lambda row: str(int(row.hex_nr))+'_'+str(int(row.point)), axis=1)
    if not ('hex_point' in df_background.keys()):
        df_background['hex_point'] = df_background.apply(lambda row: str(int(row.hex_nr))+'_'+str(int(row.point)), axis=1)
    # delta_df = df_signal.copy()
    #* merge: combine dataframes from left to right
    delta_df = pd.merge(df_signal,
                        df_background[[z_mean_col, z_err_col, 'hex_point']],
                        on='hex_point',
                        suffixes=('_s', '_bg')
                        )
    delta_df['z_mean'] = delta_df.apply(
                            lambda row: row[f'{z_mean_col}_s'] - row[f'{z_mean_col}_bg'],
                            axis=1)
    delta_df['z_err'] = delta_df.apply(
                            lambda row: quad_sum(
                                        row[f'{z_err_col}_s'],
                                        row[f'{z_err_col}_bg']
                                        ),
                            axis=1)
    delta_df.drop(columns=['hex_point'], inplace=True)
    return delta_df

#*---------------Data pipeline-------------
def preprocess_data(data_raw, precut_check=True, log_precut=True, postcut_check=True, log_postcut=False,
                    point_cut=True, mm_cut=True, median_hex_cut=True, title='title',print_removed_points=False ):
        """filters raw data & shows cotrol plots to see the distributions of points & timeseries of before
        filtering and after filtering
        returns: filter_pt: pivot table of filtered data 
        filter_df: filtered data"""
        if precut_check: 
                print('control plots precut')
                tdp.control_plots(data_raw, z_col='z', hist_log=log_precut,title=title, unit='mm')
                
    #* apply filter to dataframe
        filter_df = filter_data(data_raw,
                    point_cut=point_cut,
                    mm_cut=mm_cut, 
                    median_hex_cut=median_hex_cut,)
        
        if postcut_check: 
                print('control plots post')
                tdp.control_plots(filter_df, z_col='z', hist_log=log_postcut,title=title, unit='mm')
        # * print number of removed points
        removed_points = data_raw.z.count() - filter_df.z.count()
        if print_removed_points:
            print(f'Total points removed: {removed_points}')
            print(f'Total points removed: {removed_points/ data_raw.z.count():.2f}%')
        #* create pivot table
        filter_pt = point_table(filter_df)
        #* process pivot table further: subtract baseline & convert all units to µm
        filter_pt.z_mean = subtract_mean(filter_pt.z_mean)
        filter_pt.z_mean = convert_mm_to_microns(filter_pt.z_mean)
        filter_pt.z_err = convert_mm_to_microns(filter_pt.z_err)
        filter_pt = df_convert_unix_to_datetime(filter_pt)
        return filter_pt, filter_df
    
def data_process_pipeline(signal_raw_df, background_raw_df, print_removed_points): 
    """ function to preprocess raw signal & raw background data and to subtract them from one another
    returns: diff_pt: total dataframe of subtracted singal - background pivot tables """
    if print_removed_points: print('Background Data:')
    background_pt,_ = preprocess_data(background_raw_df,
                    precut_check=False,
                    postcut_check=False,
                    point_cut=True, mm_cut=True, median_hex_cut=True,
                    print_removed_points=print_removed_points)
    if print_removed_points: print('Signal Data:')
    singal_pt,_ = preprocess_data(signal_raw_df,
                precut_check=False,
                postcut_check=False,
                point_cut=True, mm_cut=True, median_hex_cut=True,
                print_removed_points=print_removed_points)
    diff_pt = subtract_mean_measurements(singal_pt, background_pt)
    diff_pt = add_triplet_color_label(diff_pt)
    diff_pt = add_ring_nr_label(diff_pt)
    return diff_pt

def process_curing_data(full_signal_raw_df, background_raw_df, print_removed_points=False):
    """function for applying the data_process_pipeline to a 'full_singal_raw_df' -> a dataframe which has all runs 
    concatinated in one single dataframe. It splits the full_signal_raw_df by run_nr and applies the data_process pipeline
    retuns: measurements_dict_pt -> dictonary of all subtracted data. key: run_nr_x; value: difference_pt"""
    measurements_dict_pt = {}
    for run in full_signal_raw_df.run_nr.unique(): 
        data_single_run = full_signal_raw_df.loc[full_signal_raw_df.run_nr == run, :]
        data_single_run_pt = data_process_pipeline(data_single_run, background_raw_df,
                                                    print_removed_points=print_removed_points)
        data_single_run_pt['run_nr'] = run
        measurements_dict_pt[f'run_nr_{run}'] = data_single_run_pt
    return measurements_dict_pt


def size_check_meas_dict_pt(meas_dict_pt): 
    """checks if runs are complete, removes run if points are missing"""
    return_meas_dict = {}
    run_size = meas_dict_pt['run_nr_1'].size
    for key in meas_dict_pt.keys():
        if meas_dict_pt[key].size == run_size:
            return_meas_dict[key] = meas_dict_pt[key]
    del meas_dict_pt
    return return_meas_dict


def laser_data_analysis(meas_id_sig, meas_id_bg,
                        meta_data:pd.DataFrame,
                        folder='triplets',
                        bg_data_check=False, 
                        sig_data_check=False,
                        print_removed_points=False,): 
    """function to read data by measurement_id and analyse selected data"""
    from pathlib import Path
    measurement_folder  = Path.cwd().parent / 'measurements' / folder
    #* read bg data
    bfg_filename = meta_data.loc[meta_data.measurement_id==meas_id_bg].file_name.values[0]
    print(bfg_filename)
    bfg_df = read_single_measurement(measurement_folder, bfg_filename)
    if bg_data_check:
        _,_= preprocess_data(bfg_df,
                    precut_check=True,
                    postcut_check=True,
                    point_cut=True, mm_cut=True, median_hex_cut=True, title=bfg_filename)
        
    #* read singal data
    signal_filename = meta_data.loc[meta_data.measurement_id==meas_id_sig].file_name.values[0]
    print(signal_filename)
    signal_df = read_single_measurement(measurement_folder, signal_filename)
    if sig_data_check:
        _,_ = preprocess_data(signal_df,
                precut_check=True,
                postcut_check=True,
                point_cut=True, mm_cut=True, median_hex_cut=True, title=signal_filename)
    #* process both datasets
    meas_dict_pt = process_curing_data(signal_df, bfg_df, print_removed_points=print_removed_points)
    meas_dict_pt = size_check_meas_dict_pt(meas_dict_pt)
    return meas_dict_pt


#*-----------------------------------------
def subtract_mean(data):
    mean_data = np.mean(data)
    data = data.apply(lambda z: (z - mean_data))
    return data

def convert_mm_to_microns(data):
    data = data.apply(lambda x: x*1e3)
    return data
    
def calc_min_mean_max(dataframe, mode="z_mean"):
    return dataframe[mode].min(), dataframe[mode].mean(), dataframe[mode].max() 

def calc_normalize_value(dataframe, mode='z'):
    mode_mean = dataframe[mode].mean()
    mode_sig = dataframe[mode].std()
    dataframe[f'{mode}_norm'] = (dataframe[mode] - mode_mean) / mode_sig
    return dataframe

def square_sum(array): 
    return np.sqrt(np.sum(np.power(array,2)))

def calc_R_from_data(data):
    """Helper function for calc_flats_statistics, returns Range and RangeErr per 
       measurements_dict_pt entry"""
    z_min = data.z_mean.min()
    z_max = data.z_mean.max()
    min_err = data.loc[data.z_mean==z_min].z_err.values[0]
    max_err = data.loc[data.z_mean==z_max].z_err.values[0]
    R = z_max - z_min
    deltaR = np.sqrt(square_sum([min_err, max_err]))
    return R, deltaR

def calc_flats_statistic_df(measurements_dict_pt: dict):
    '''Calculates summary statistic dataframe from measurements_dict_pt 
       measurements_dict_pt is the dict of data of curing per run (see examples)'''
    Rs = []
    deltaRs = []
    RMSs = []
    time = []
    for key, data in measurements_dict_pt.items(): 
        R, dR = calc_R_from_data(data)
        Rs.append(R)
        deltaRs.append(dR) 
        
        RMS = data.z_mean.std()
        RMSs.append(RMS)
        
        mean_time = data.unix_time.mean()
        time.append(mean_time)
    flats_statistic_df = pd.DataFrame({'R': Rs, 'deltaR': deltaRs, 'RMS': RMSs, 'unix_time': time})
    flats_statistic_df['run_nr'] = [int(run_nr.split('_')[2]) for run_nr in measurements_dict_pt.keys()]

    flats_statistic_df = df_convert_unix_to_datetime(flats_statistic_df)
    flats_statistic_df = add_column_time_passed(flats_statistic_df)
    flats_statistic_df['odd_runs'] = flats_statistic_df.run_nr.apply(lambda x: int(x)%2)
    return flats_statistic_df

def calc_measurement_date_and_time(dataframe, timemode='unix_time', time_format='%H:%M - %d.%m.%Y'): 
    '''calcs mean time of measurement,
       timemode column needs unix timestamp
       return: string of mean time of the measurement'''
    from datetime import datetime
    mean_time = dataframe[timemode].mean()
    mean_datetime = pd.to_datetime(mean_time, unit='s', utc=True)
    mean_datetime.tz_convert('Europe/Berlin')
    str_datetime = mean_datetime.strftime(format=time_format)
    return str_datetime

# ------------- microscope helpers ------------

def find_distance(row, df_coords): 
    """returns dict with adjecent points label & coordinates for a row 
       used as helper function for add_adjecent_points"""
    distances = {'adj_x1':None, 'adj_x2':None, 'adj_y1':None , 'adj_y2':None,
                 'dist_1':None, 'dist_2':None, 'label_1':None, 'label_2':None}
    point_distance = 20 #mm
    def distance(x1,y1,x2,y2): return np.sqrt((x1-x2)**2 + (y1-y2)**2) # euclidian distance
    i = 1
    for coord_row in df_coords.itertuples(): 
        d = distance(row.x, row.y, coord_row.x, coord_row.y)
        if 1 < d < point_distance:
            distances[f'adj_x{i}'] = coord_row.x
            distances[f'adj_y{i}'] = coord_row.y
            distances[f'dist_{i}'] = d
            distances[f'label_{i}'] = coord_row.arm_label
            i = i+1
    return distances

def add_adjecent_points(df, df_coords):
    """returns dataframe with adjecent points label and coordinates (from  df_coords) for each arm_label in df
       used for microscope data visualisation """
    merge_df = pd.DataFrame(columns=['arm_label', 'adj_x1', 'adj_y1', 'dist_1', 'adj_x2', 'adj_y2', 'dist_2'])
    for row in df.itertuples():
        distances = find_distance(row, df_coords)
        distances['arm_label'] = row.arm_label
        distances = pd.DataFrame([distances])
        merge_df = pd.concat([merge_df, distances], ignore_index=True)
    return merge_df