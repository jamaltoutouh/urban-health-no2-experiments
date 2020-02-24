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
import sys
import glob
import sys
from scipy.stats import shapiro


station = '035'
metric = '10'


data_txt = '../data/txt/'
data_csv = '../data/csv/'
root_path = '.'
os.listdir(root_path)


def get_season(day, month):
    if (day >= 21 and month == 3) or month == 4 or month == 5 or (day < 21 and month == 6):
        return 'spring'
    elif (day >= 21 and month == 6) or month == 7 or month == 8 or (day < 21 and month == 9):
        return'summer'
    elif (day >= 21 and month == 9) or month == 10 or month == 11 or (day < 21 and month == 12):
        return 'autumn'
    elif (day >= 21 and month == 12) or month == 1 or month == 2 or (day < 21 and month == 3):
        return 'winter'
    else:
        return 'error'


def get_working_day(week_day):
    return week_day != 5 and week_day != 6


def get_sequence(data_df):
    hours = data_df.columns[0:24]
    list_of_data = []
    for index, row in data_df.iterrows():
        for hour in hours:
            data = {}
            data['hour'] = int(hour)
            data['month'] = row['month']
            data['year'] = row['year']
            data['day'] = row['day']
            data['date'] = datetime(row['year'], row['month'], row['day'], data['hour'], 0, 0)
            data['measure'] = row[hour]
            data['weekday'] = row['weekday']
            data['season'] = get_season(row['day'], row['month'])
            data['working-day'] = get_working_day(row['weekday'])

            list_of_data.append(data)

    return pd.DataFrame(list_of_data)


if len(sys.argv) < 2:
    print('Error: Arguments required')
    print('Run: python transform-data-to-sequence.py <station_id> <metric_id>')
    print('For example (Madrid Central and NO2): python transform-data-to-sequence.py 035 08')
else:
    station = sys.argv[1]
    metric = sys.argv[2]
    data_df = pd.read_csv(data_csv + station + '-' + metric + '-air_all.csv')
    get_sequence(data_df).sort_values(by=['date']).to_csv(data_csv + station + '-' + metric + '-sequence_air_all.csv', index=False)