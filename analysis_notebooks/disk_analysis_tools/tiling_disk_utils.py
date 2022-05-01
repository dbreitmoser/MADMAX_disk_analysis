import pandas as pd
import numpy as np 
from scipy.stats import median_abs_deviation as mad  
mad_str = 'median_abs_deviation'
import pytz
local_tz = pytz.timezone('Europe/Berlin')


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
# #! outdated? 
# def apply_func_per_run(dataframe, func, pivot=False): 
#     return_df = pd.DataFrame()
#     for run in dataframe.run_nr.unique(): 
#         run_df = dataframe.loc[dataframe['run_nr']==run, :].copy()
#         run_df = func(run_df)
#         if pivot: run_df['run_nr'] = run
#         return_df = pd.concat([return_df, run_df])
#     del dataframe
#     return return_df
# #!---

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
        df['z_clean'] = df[mode].apply(lambda z: temp_sort_func(z))
    return df 

#! OUTDATED 
# def hex_table(dataframe):
#     """returns DF pivot table with mean, std and median of z by hexagon"""
#     hex_table = pd.pivot_table(dataframe, values=['x','y','z'], index=['hex_nr'],
#                             aggfunc={
#                                 'x': np.mean,
#                                 'y': np.mean,
#                                 'z': [np.mean, measurement_error,]})
#     hex_table[('x','mean')] = hex_table[('x','mean')].apply(lambda x: round(x,2))
#     hex_table[('y','mean')] = hex_table[('y','mean')].apply(lambda x: round(x,2))
#     hex_table.columns = ['_'.join(col).rstrip('_') for col in hex_table.columns.values]
#     hex_table.rename(columns = {'x_mean':'x', 'y_mean':'y'}, inplace = True)
#     return hex_table

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
    temp_humid_db = pd.DataFrame(t, columns=["date",
                         "T_plate",
                         "T_ambient",
                         "humidity", 
                         "dewpoint", 
                         "fan_speed"])
    temp_humid_db['datetime'] = temp_humid_db['date'].apply(lambda x: pd.to_datetime(x))
    temp_humid_db['datetime'].dt.tz_localize('Europe/Berlin')
    return temp_humid_db


def df_convert_unix_to_datetime(df: pd.DataFrame):
    """Takes dataframe with unix timestamp and replaces by python datetime object + converts to berlin timezone.
       new column: 'date' -> datetime.datetime object (UTC +2 europe/berlin) """
       
    df['datetime'] = df['unix_time'].apply(lambda x: pd.to_datetime(x, unit='s', utc=True))
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/Berlin')
    df.drop('unix_time', axis=1)
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

def difference_data(df_1, df_2): 
    '''Takes only pivot df from the point table or hex table function!'''    
    delta_df = df_1 - df_2
    delta_df['x', 'mean'] = df_1['x', 'mean'] 
    delta_df['y', 'mean'] = df_1['y', 'mean'] 
    delta_df['z', 'std'] = quad_sum(df_1['z', 'std'], df_2['z', 'std'])
    return delta_df

def z_diff_raw_data(df_1, df_2): 
    '''Returns measurement dataframe with z_new = df_1.z - df_2.z'''
    delta_df = df_1 
    delta_df['z'] = df_1['z'] - df_2['z']
    return delta_df


def combine_mean_measurements(data_signal, data_background, z_mean_col='z_mean', z_err_col='z_err'):
    # Takes only point_tables
    delta_df = data_signal.copy()
    delta_df[z_mean_col] = data_signal[z_mean_col] - data_background[z_mean_col]

    if 'unix_time' in data_signal.keys(): 
    #     delta_df['datetime'] = data_signal['datetime']
        delta_df['unix_time'] = data_signal['unix_time']
    delta_df[z_err_col] = quad_sum(data_signal[z_err_col], data_background[z_err_col])
    return delta_df 


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
    flats_statistic_df = df_convert_unix_to_datetime(flats_statistic_df)
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