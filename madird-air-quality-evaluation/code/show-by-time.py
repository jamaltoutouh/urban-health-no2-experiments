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
data_path = '../data/csv/035-08-sequence_air_all.csv'
hours = list(range(24))


data = pd.read_csv(data_path)
print(data.columns)

day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# For the first graph
week_data = dict()
for i, day  in enumerate(day_name):
    week_data[day] = []
    for j in range(24):
        measures = list(data[(data['weekday'] == i) & (data['hour'] == j)]['measure'])
        week_data[day].append(mean(measures))

# For the second graph
workingday_data = dict()
hours_rev = hours[::-1]
for is_working in [True, False]:
    label = 'Working day' if is_working else 'Weekend'
    workingday_data[label] = dict()
    workingday_data[label]['mean'] = []
    workingday_data[label]['max'] = []
    workingday_data[label]['min'] = []
    for j in range(24):
        measures = list(data[(data['working-day'] == is_working) & (data['hour'] == j)]['measure'])
        sstedv = stdev(measures)
        mmean = mean(measures)
        workingday_data[label]['mean'].append(mmean)
        workingday_data[label]['max'].append(mmean + sstedv)
        workingday_data[label]['min'].append(mmean - sstedv)
    workingday_data[label]['min'] = workingday_data[label]['min'][::-1]

dashes = ['-', '--', '-.', ':']
w, h = figaspect(3 / 6)
f, ax = plt.subplots(figsize=(w, h))
sns.set(style="whitegrid")
ax.set_ylabel('NO$_2$ concentration', )
x = range(24)
plots = []
for i, day in enumerate(day_name):
    if i < 5:
        color = (0, 0, (255 - 40 * i)/255)
    else:
        color= ((255-40*(i%5))/255, 0, 0)


    plots.append(ax.plot(x, week_data[day], label=day, color=color, linestyle=dashes[i%4],))
ax.set_xlabel(r"Hour of the day", fontsize = 16)
ax.set_ylabel(r'NO2 concentration', fontsize = 16)
ax.tick_params(labelsize=14)
# always call tight_layout before saving ;)
plt.tight_layout()
plt.xticks(range(24))
plt.margins(x=0)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.savefig('no2_concentration_hour.png')
plt.show()
