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

from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import iqr
import scikit_posthocs as sp

data_path = '../data/csv/sequence_air_all.csv'

dataset = pd.read_csv(data_path, header=0, index_col=0)

df_pre_mc = dataset[(dataset.index >= "2017-12-01") & (dataset.index < "2018-09-31")]
df_post_mc = dataset[(dataset.index >= "2018-12-01") & (dataset.index < "2019-09-31")]
day_name = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

print(dataset.columns)
df_pre_mc['period'] = 'Without pedestrianization'
df_post_mc['period'] = 'With pedestrianization'

data = pd.concat([df_pre_mc, df_post_mc])
data.to_csv('../data/csv/data_for_comparing_by_time.csv')

data_to_plot = {}
for day in range(7):
    gaps = []
    for h in range(24):
        print('{} - {}'.format(day, h))
        pre = np.array(df_pre_mc[(df_pre_mc['hour'] == h) & (df_pre_mc['weekday'] == day)]['measure'])
        post = np.array(df_post_mc[(df_post_mc['hour'] == h) & (df_post_mc['weekday'] == day)]['measure'])
        #print(pre)
        gaps.append(pre.mean()-post.mean())
    data_to_plot[day] = gaps

w, h = figaspect(3 / 6)
f, ax = plt.subplots(figsize=(w, h))
sns.set(style="whitegrid")
ax.set_ylabel('NO$_2$ concentration', fontweight='bold')
x = range(24)
plots = []
for day in range(7):
    plots.append(ax.plot(x, data_to_plot[day], label=day_name[day]))
ax.set_xlabel(r"Hour of the day", fontweight='bold', fontsize = 16)
ax.set_ylabel(r'gap', fontweight='bold', fontsize = 16)
ax.tick_params(labelsize=14)
# always call tight_layout before saving ;)
plt.tight_layout()
plt.xticks(range(24))
plt.margins(x=0)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('comparing_no2_concentration_hour.png')
plt.show()

sys.exit(0)

day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
min_len = 99999999
week_data = {}
for i, day  in enumerate(day_name):
    week_data[day] = []
    for j in range(24):
        measures = list(data[(data['weekday'] == i) & (data['hour'] == j)]['measure'])
        week_data[day].append(mean(measures))


w, h = figaspect(3 / 6)
f, ax = plt.subplots(figsize=(w, h))
sns.set(style="whitegrid")
ax.set_ylabel('NO$_2$ concentration', fontweight='bold')
x = range(24)
plots = []
for day in day_name:
    plots.append(ax.plot(x, week_data[day], label=day))
ax.set_xlabel(r"Hour of the day", fontweight='bold', fontsize = 16)
ax.set_ylabel(r'NO2 concentration', fontweight='bold', fontsize = 16)
ax.tick_params(labelsize=14)
# always call tight_layout before saving ;)
plt.tight_layout()
plt.xticks(range(24))
plt.margins(x=0)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('no2_concentration_hour.png')
plt.show()
