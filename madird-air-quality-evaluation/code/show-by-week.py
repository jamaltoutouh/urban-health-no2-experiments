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

data = pd.read_csv(data_path)
print(data.columns)

day_name = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
min_len = 99999999
week_data = {}
for i, day  in enumerate(day_name):
    week_data[day] = list(data[data['weekday'] == i]['measure'])
    print(len(week_data[day]))
    if min_len > len(week_data[day]):
        min_len = len(week_data[day])

# Igualamos los arrays para crear el DF
for i, day  in enumerate(day_name):
    week_data[day] = week_data[day][:10824]

week_df = pd.DataFrame(week_data)


w, h = figaspect(3 / 8)
f, ax = plt.subplots(figsize=(w, h))
sns.set(style="whitegrid")
ax.set_ylabel('NO$_2$ concentration', fontweight='bold')
ax = sns.boxplot(data=week_df[day_name],  showfliers=False)
plt.margins(x=0)
plt.savefig('no2_concentration_week.png')
plt.show()
