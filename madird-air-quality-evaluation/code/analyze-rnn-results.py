import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import figaspect
import seaborn as sns
import glob

import scipy.stats
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro
from statistics import mean

from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import iqr
import scikit_posthocs as sp
import os

resutls_folder = '.'

def get_folders(hiden_layer_size, loock_back):
    return glob.glob(resutls_folder + '/*' + str(hiden_layer_size) + '-' + str(loock_back))


def get_results_from_csv(folder, csv_file):
    if not os.path.exists(folder + '/' + csv_file): return None
    data = pd.read_csv(folder + '/' + csv_file)
    predicted = np.array(data['predicted'].tolist())
    actual = np.array(data['actual'].tolist())

    over_predicted_ratio = sum(actual < predicted) / len(predicted)

    diff = predicted - actual
    over_predicted_val = np.sum(diff[diff>0]) / np.sum(diff>0)
    over_predicted_ratio = np.sum(diff>0) / len(predicted)

    mse = sum(data['mse'].tolist()) / len(predicted)
    mae = sum(data['mae'].tolist()) / len(predicted)

    return mse, mae, over_predicted_ratio, over_predicted_val


def get_results_pre_MC(folder):
    return get_results_from_csv(folder, 'pre-MC.csv')

def get_results_post_MC(folder):
    return get_results_from_csv(folder, 'post-MC.csv')


def get_results(hiden_layer_size, loock_back):
    folders = get_folders(hiden_layer_size, loock_back)
    for folder in folders:
        results_pre_MC = get_results_pre_MC(folder)
        if not results_pre_MC is None:
            print('Pre-MC')
            print(results_pre_MC)

        results_post_MC = get_results_post_MC(folder)
        if not results_post_MC is None:
            print('Post-MC')
            print(results_post_MC)



hidden_layer_size = [2, 4, 6, 8, 10, 100]
loock_backs = [6, 12, 24]

for hls in hidden_layer_size:
    for lb in loock_backs:
        print('{} - {}'.format(hls, lb))
        get_results(hls, lb)