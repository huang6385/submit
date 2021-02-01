import re 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def parse_string_re(s1,s2,split_sign = '{}'):
    s_list = s1.split(split_sign)
    while '' in s_list:
        s_list.remove('')
    score = []
    for s in s_list:
        s_re = re.compile(s + '[+-]?((\d+(\.\d*)?)|(\.\d+))')
        smath = re.search(s_re,s2)
        num = smath.group(0).replace(s,'')
        score.append(float(num))
    return score


def parse_string(s1,s2):
    '''
    s1 is re_s
    s2 
    like follow:
    s1 = 'dasjkl(),asdkjqwkjelqwe()asdjklqwe'
    s2 = 'dasjkl(1.2),asdkjqwkjelqwe(3.5)asdjklqwe'

    '''

    i = 0
    j = 0
    res = []
    s = ''
    sign = False
    while i<len(s1) and j<len(s2):
        if s1[i] == s2[j]:
            if sign:
                res.append(s)
                s = ''
                sign = False
            i+=1
            j+=1
        else:
            sign = True
            s = s + s2[j]
            j += 1
    if s!='':res.append(s)
    return res




def vis():
    


    pickle_file = 'data/sensor_graph/adj_mx.pkl'
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    
    width = 128
    hidth = 90
    sensor_dis = np.zeros((hidth,width))
    sensor_counter = np.zeros((hidth,width))

    station_loc = pd.read_csv('data/sensor_graph/graph_sensor_locations.csv')

    min_lat = np.min(station_loc['latitude'])
    max_lat = np.max(station_loc['latitude'])
    lat_unit = (max_lat-min_lat + 1e-5)/hidth
    min_lon = np.min(station_loc['longitude'])
    max_lon = np.max(station_loc['longitude'])
    lon_unit = (max_lon-min_lon + 1e-5)/width


    # all station node distribution
    idx2lat_lon = {}
    for idx,lat,long in zip(station_loc['index'],station_loc['latitude'],station_loc['longitude']):
        lat_idx = int((lat-min_lat)/lat_unit)
        lon_idx = int((long-min_lon)/lon_unit)
        idx2lat_lon[idx] = [lat_idx,lon_idx]

        sensor_dis[lat_idx,lon_idx] += 1
    return hidth,width,idx2lat_lon,sensor_dis