import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 13
import seaborn as sns


data_txt = '../data/txt/'
data_csv = '../data/csv/'

dataset = pd.read_csv(data_csv+'sequence_air_all.csv', header=0, index_col=0)
measures = dataset['measure']

sns.set(style="whitegrid")
sns.set_style("ticks")
w, h = figaspect(4 / 12)
f, ax = plt.subplots(figsize=(w, h))
x = range(len(measures))
plot_bce = ax.plot(x, measures, 'b-')
# no plot without labels on the axis
ax.set_xlabel(r"Training epoch", fontweight='bold', fontsize = 16)
ax.set_ylabel(r'NO2 concentration', fontweight='bold', fontsize = 16)
ax.tick_params(labelsize=14)
# always call tight_layout before saving ;)
plt.tight_layout()
#fig.savefig("../../experimental-data-TELO2019/data/images/mnist-3x3_fid-evolution-all.eps", dpi=300)
plt.show()



#
# # specify columns to plot
# groups = [0, 1, 2, 3, 5, 6, 7]
# i = 1
# # plot each column
# pyplot.figure()
# for group in groups:
# 	pyplot.subplot(len(groups), 1, i)
# 	pyplot.plot(values[:, group])
# 	pyplot.title(dataset.columns[group], y=0.5, loc='right')
# 	i += 1
# pyplot.show()