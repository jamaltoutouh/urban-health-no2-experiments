import string
import re
import json
import pandas as pd
from datetime import datetime
import numpy as np
import math

from collections import OrderedDict
from datetime import date

import os
import glob
import sys
from scipy.stats import shapiro

data_txt = '../data/txt/'
data_csv = '../data/csv/'
root_path = '.'


def get_csv_files(path):
    return [file_path for file_path in glob.iglob(path + 'air_*.csv')]

def get_txt_files(path):
    return [file_path for file_path in glob.iglob(path + 'air_*.txt')]

def get_measures(data, line):
    for i in range(24):
        if 'V' in line[(i*2)+1]:
            data['{:02d}'.format(i)] = float(line[i*2])
        else:
            data['{:02d}'.format(i)] = float(-1)
    return data

def get_measures_txt(data, line):
    for i in range(24):
        j=i*6+4+1
        if 'V' in line[j]:
            data['{:02d}'.format(i)] = float(line[i*6:j])
        else:
            data['{:02d}'.format(i)] = float(-1)
    return data

def parse_csv_files(files, station, metric):
    list_of_data = []
    for file_name in files:
        for line in open(file_name, 'r'):
            splited_line = re.split(",", line)
            if station in splited_line[2] and metric in splited_line[3]:

                data = {}
                data['station'] = station
                data['metric'] = metric
                data['year'] = int(splited_line[6])
                data['month'] = int(splited_line[7])
                data['day'] = int(splited_line[8])
                data['weekday'] = datetime(data['year'],data['month'],data['day']).weekday()
                data['date'] = datetime(data['year'],data['month'],data['day'])
                data = get_measures(data, splited_line[9:])
                list_of_data.append(data)

    return pd.DataFrame(list_of_data)


def parse_txt_files(files, station, metric):
    list_of_data = []
    for file_name in files:
        for line in open(file_name, 'r'):

            if station+metric in line:
                data = {}
                data['station'] = station
                data['metric'] = metric
                data['year'] = int('20'+line[14:16])
                data['month'] = int(line[16:18])
                data['day'] = int(line[18:20])
                data['weekday'] = datetime(data['year'],data['month'],data['day']).weekday()
                data['date'] = datetime(data['year'], data['month'], data['day'])
                data = get_measures_txt(data, line[20:])
                list_of_data.append(data)

    return pd.DataFrame(list_of_data)


# Metrics:
# - 08: NO2
# - 10: PM10


if len(sys.argv) < 2:
    print('Error: Arguments required')
    print('Run: python extract-information-from-txt.py <station_id> <metric_id>')
    print('For example (Madrid Central and NO2): python transform-data-to-sequence.py 035 08')
else:
    station = sys.argv[1]
    metric = sys.argv[2]

files = get_txt_files(data_txt)
print(files)
data_df_txt = parse_txt_files(files, station, metric)

files = get_csv_files(data_txt)
print(files)
data_df_csv = parse_csv_files(files, station, metric)
data_df = pd.concat([data_df_txt, data_df_csv])
print(data_df)

for h in range(24):
    col = '{:02d}'.format(h)
    data_df.loc[(data_df[col] == -1), col] = np.array(data_df[col]).mean()
    print(np.array(data_df[col]).mean())

data_df.sort_values(by=['year', 'month', 'day']).to_csv(data_csv + station + '-' + metric + '-air_all.csv', index=False)
