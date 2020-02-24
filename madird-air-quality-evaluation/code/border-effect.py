import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 16
from matplotlib.figure import figaspect
import seaborn as sns

import scipy.stats
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro
from statistics import mean
from random import sample

from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import iqr
import scikit_posthocs as sp


if len(sys.argv) < 3:
    print('Error: Arguments required')
    print('Run: python transform-data-to-sequence.py <station_id> <metric_id> <table_type> [include_seasons]')
    print('<table_type>: gap, gap-percentage, both-gaps or time-lower-threshold')
    print('For example (Madrid Central and NO2 and gap): python transform-data-to-sequence.py 035 08 spring')
    sys.exit(0)
else:
    station = sys.argv[1]
    metric = sys.argv[2]
    table_type = sys.argv[3]  # 'gap', 'gap-percentage', 'time-lower-threshold'
    include_seasons = sys.argv[4].lower() == 'true'
    data_path = '../data/csv/' + station + '-' + metric + '-sequence_air_all.csv'



dataset = pd.read_csv(data_path, header=0, index_col=0)

whole_pre = dataset[(dataset.index >= "2014-03-19") & (dataset.index <= "2018-11-30")]
whole_post = dataset[(dataset.index >= "2018-12-01") & (dataset.index <= "2019-11-30")]





def normality_test(algorithm, data, metric_title, alpha):
    print(data)
    stat, p = shapiro(data)
    stats_string = '{}\t {}\t {}\t {}\t {}'.format(algorithm, metric_title, p, stat, p<alpha)
    print(stats_string)


def pairwise_test(algorithm1, data1, algorithm2, data2, metric_title, alpha):
    t, p = scipy.stats.f_oneway(data1, data2)
    #t, p = wilcoxon(data1, data2[0:12])
    stats_string = '{} vs {}\t {}\t {}\t {}\t {}'.format(algorithm1, algorithm2, metric_title, p < alpha, p, t)
    #print(stats_string)
    return t, p, p<alpha


def get_stats(data_list, threshold):
    val = np.array(data_list)
    mean = val.mean()
    minn = val.min()
    maxx = val.max()
    norm_stdev = val.std()/mean * 100
    time_bellow_threshold = sum(val < threshold)/len(val)
    return minn, mean, norm_stdev, maxx, time_bellow_threshold


def get_df_working_day_working_day(data):
    data_dict = dict()
    is_working = True
    data_dict['working-day'] = list(data[data['working-day'] == is_working]['measure'])
    is_working = False
    data_dict['weekend'] = list(data[data['working-day'] == is_working]['measure'])
    return data_dict

stations = pd.read_csv('../stations-information.csv', header=0, index_col=0)
station_name = (list(stations[stations.index == int(station)]['Estacion-Latex'])[0])
stat_diff_string = '\STATDIFF'

gaps = dict()
gaps['working-day'] = []
gaps['weekend'] = []

means = dict()
means['working-day'] = []
means['weekend'] = []

time_bellow_pre = dict()
time_bellow_pre['working-day'] = []
time_bellow_pre['weekend'] = []
time_bellow_post = dict()
time_bellow_post['working-day'] = []
time_bellow_post['weekend'] = []

seasons = ['spring', 'summer', 'autumn', 'winter']

string = station_name + ' ' if include_seasons else ' '

for season in seasons:
    df_pre_mc = whole_pre[(whole_pre.season == season)]
    df_post_mc = whole_post[(whole_post.season == season)]

    dict_pre_mc = get_df_working_day_working_day(df_pre_mc)
    dict_post_mc = get_df_working_day_working_day(df_post_mc)

    day_type = ['working-day', 'weekend']
    threshold = 40
    string += '&'
    for i, day in enumerate(day_type):
        pre = list(dict_pre_mc[day])
        post = list(dict_post_mc[day])
        t, p, res = pairwise_test('Pre-MC', pre, 'Post-MC', post, 'NO$_2$', 0.01)
        pre_min, pre_mean, pre_stdev, pre_max, pre_test = get_stats(pre, threshold)
        post_min, post_mean, post_stdev, post_max, post_test = get_stats(post, threshold)
        gap = post_mean - pre_mean
        gaps[day].append(gap)
        means[day].append(pre_mean) # For the percentage
        time_bellow_pre[day].append(pre_test)
        time_bellow_post[day].append(post_test)

        test1 = stat_diff_string if not res else ''
        test2 = '' if gap>0 else ''

        if include_seasons:
            if table_type == 'gap':
                string += '& {} {} {:.2f} '.format(test1, test2, gap)
            elif table_type == 'gap-percentage':
                string += '& {} {} {:.2f}\\% '.format(test1, test2, gap/pre_mean*100)

df_pre_mc = whole_pre
df_post_mc = whole_post


means['working-day'] = [g/m for g, m in zip(gaps['working-day'], means['working-day'])]
means['working-day'] = sum(means['working-day'])/len(means['working-day'])
means['weekend'] = [g/m for g, m in zip(gaps['weekend'], means['weekend'])]
means['weekend'] = sum(means['weekend'])/len(means['weekend'])

gaps['working-day'] = sum(gaps['working-day'])/len(gaps['working-day'])
gaps['weekend'] = sum(gaps['weekend'])/len(gaps['weekend'])

dict_pre_mc = get_df_working_day_working_day(df_pre_mc)
dict_post_mc = get_df_working_day_working_day(df_post_mc)

if not include_seasons: string = station_name + ' '
string += '&'
for i, day in enumerate(day_type):
    pre = list(dict_pre_mc[day])
    post = list(dict_post_mc[day])
    t, p, res = pairwise_test('Pre-MC', pre, 'Post-MC', post, 'NO$_2$', 0.01)
    pre_min, pre_mean, pre_stdev, pre_max, pre_test = get_stats(pre, threshold)
    post_min, post_mean, post_stdev, post_max, post_test = get_stats(post, threshold)
    gap = post_mean - pre_mean

    test1 = stat_diff_string if not res else '  '
    test2 = ' ' if gap > 0 else '  '

    if table_type == 'gap':
        string += '& {} {} {:.2f} '.format(test1, test2, gaps[day])
    elif table_type == 'gap-percentage':
        string += '& {} {} {:.2f}\\% '.format(test1, test2, means[day]*100)
    elif table_type == 'both-gaps':
        string += '& {} {} {:.2f} '.format(test1, test2, gaps[day])
        string += '& {:.2f}\\% '.format(means[day] * 100)

if table_type == 'gap':
    global_gap = (gaps['working-day'] * 5 + gaps['weekend'] * 2) / 7
    string += ' & {:.2f} \\\  % {:.2f}'.format(global_gap, global_gap)
elif table_type == 'gap-percentage':
    global_mean = (means['working-day'] * 5 + means['weekend'] * 2) / 7
    string += ' & {:.2f}\\% \\\  % {:.2f}'.format(global_mean * 100, global_mean * 100)
elif table_type == 'both-gaps':
    global_gap = (gaps['working-day'] * 5 + gaps['weekend'] * 2) / 7
    string += ' & {:.2f}  '.format(global_gap, global_gap)
    global_mean = (means['working-day'] * 5 + means['weekend'] * 2) / 7
    string += ' & {:.2f}\\% \\\  % {:.2f}'.format(global_mean * 100, global_mean * 100)


if table_type == 'time-lower-threshold':
    for i, day in enumerate(day_type):
        time_bellow_pre[day] = mean(time_bellow_pre[day])
        time_bellow_post[day] = mean(time_bellow_post[day])
    global_time_bellow_pre = (time_bellow_pre['working-day'] * 5 + time_bellow_pre['weekend'] * 2) / 7
    global_time_bellow_post = (time_bellow_post['working-day'] * 5 + time_bellow_post['weekend'] * 2) / 7

    string += '& {:.2f}\\% & {:.2f}\\% & {:.2f}\\% '.format(time_bellow_pre['working-day'] * 100,
                                                                        time_bellow_pre['weekend'] * 100,
                                                                        global_time_bellow_pre*100)
    string += '&& {:.2f}\\% & {:.2f}\\% & {:.2f}\\%  '.format(time_bellow_post['working-day'] * 100,
                                                                        time_bellow_post['weekend'] * 100,
                                                                        global_time_bellow_post*100)
    string += ' && {:.2f}\\% \\\ '.format(100*(global_time_bellow_post-global_time_bellow_pre))


print(string)

sys.exit(0)
