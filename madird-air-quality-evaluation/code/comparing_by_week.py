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
    print('Run: python transform-data-to-sequence.py <station_id> <metric_id> <season>')
    print('For example (Madrid Central and NO2 and spring): python transform-data-to-sequence.py 035 08 spring')

    sys.exit(0)
else:
    station = sys.argv[1]
    metric = sys.argv[2]
    season = sys.argv[3]
    data_path = '../data/csv/' + station + '-' + metric + '-sequence_air_all.csv'



dataset = pd.read_csv(data_path, header=0, index_col=0)

df_pre_mc = dataset[(dataset.index >= "2011-12-01") & (dataset.index < "2018-09-31") & (dataset.season == season)]
df_post_mc = dataset[(dataset.index >= "2018-12-01") & (dataset.index < "2019-09-31") & (dataset.season == season)]


day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def get_df_week_week(data):
    min_len = 99999999
    week_data = {}
    for i, day  in enumerate(day_name):
        week_data[day] = list(data[data['weekday'] == i]['measure'])
        print(len(week_data[day]))
        if min_len > len(week_data[day]):
            min_len = len(week_data[day])

    # Igualamos los arrays para crear el DF
    for i, day  in enumerate(day_name):
        week_data[day] = sample(week_data[day], min_len)

    return pd.DataFrame(week_data)


#df_pre_mc = get_df_week_week(df_pre_mc)
df_pre_mc['period'] = 'Without pedestrianization'
#week_post_mc = get_df_week_week(df_post_mc)
df_post_mc['period'] = 'With pedestrianization'

data = pd.concat([df_pre_mc, df_post_mc])
data.to_csv('../data/csv/data_for_boxplot.csv')
print(data)




w, h = figaspect(3 / 8)
f, ax = plt.subplots(figsize=(w, h))
sns.set(style="whitegrid")
ax.set_ylabel('NO$_2$ concentration', fontweight='bold')
#ax = sns.boxplot(data=week_pre_mc[day_name],  showfliers=False)
#ax = sns.boxplot(data=week_post_mc[day_name],  showfliers=False)
sns.boxplot(y='measure', x='weekday',  hue='period', data=data, showfliers=False).set(
    xlabel='',
    ylabel='NO$_2$ concentration'
)

ax.set_xticklabels(day_name)
plt.margins(x=0)
plt.savefig(station + '-' + metric + '-' + season + '-comparison_no2_concentration_week.png')
#plt.show()


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


def print_noise_stats(noise_stats, median = True):
    if not median:
        result_string = '\multirow{2}{*}{' + noise_stats.iloc[0]['metric'] + '}  '
        result_string += '&  {} & {:.2f} & {:.2f}$\pm${:.2f}\% & {:.2f}  & \n'.format(noise_stats.iloc[0]['period'], noise_stats.iloc[0]['min'], noise_stats.iloc[0]['mean'], noise_stats.iloc[0]['norm_stdev'], noise_stats.iloc[0]['max'])
        result_string += '\multirow{2}{*}{' + '{:.2f}\%'.format(noise_stats.iloc[1]['gap']) + '} \\\\  \n'
        result_string += '& {} & {:.2f} & {:.2f}$\pm${:.2f}\% & {:.2f}   \\\\'.format(noise_stats.iloc[1]['period'],
                                                                                     noise_stats.iloc[1]['min'],
                                                                                     noise_stats.iloc[1]['mean'],
                                                                                     noise_stats.iloc[1]['norm_stdev'],
                                                                                     noise_stats.iloc[1]['max'])


def get_stats(data_list):
    val = np.array(data_list)
    mean = val.mean()
    minn = val.min()
    maxx = val.max()
    norm_stdev = val.std()/mean * 100

    return minn, mean, norm_stdev, maxx



df_pre_mc = get_df_week_week(df_pre_mc)
df_post_mc = get_df_week_week(df_post_mc)

for i, day in enumerate(day_name):
    pre = list(df_pre_mc[day])
    post = list(df_post_mc[day])
    #print(normality_test('Pre-MC', pre, 'NO$_2$', 0.01))
    #print(normality_test('Post-MC', post, 'NO$_2$', 0.01))
    t, p, res = pairwise_test('Pre-MC', pre, 'Post-MC', post, 'NO$_2$', 0.01)
    stats_string = '{} & {:.3f} & {:.3f} & {}'.format(day, t, p, res)
    #print(stats_string)
    pre_min, pre_mean, pre_stdev, pre_max = get_stats(pre)
    post_min, post_mean, post_stdev, post_max = get_stats(post)
    gap = post_mean - pre_mean
    stats = '{} & {:.2f} & {:.2f}$\\pm${:.2f}\\% & {:.2f} & {:.2f} & {:.2f}$\\pm${:.2f}\\% & {:.2f} & {:.2f} \\\ '.format(
        day, pre_min, pre_mean, pre_stdev, pre_max, post_min, post_mean, post_stdev, post_max, gap)
    print(stats)
