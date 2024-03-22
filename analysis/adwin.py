import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from river import drift

def read_csv(epsilon_sep, epsilon_dec):

    sep_data = pd.read_csv(epsilon_sep)
    dec_data = pd.read_csv(epsilon_dec)

    return sep_data["epsilon"], dec_data["epsilon"]

def extract(sep_epsilons, dec_epsilons):

    diff_eps = []

    for sep_eps, dec_eps in zip(sep_epsilons, dec_epsilons):
    
        diff_eps.append(abs(sep_eps-dec_eps))

    print(len(diff_eps), "len differences")

    return diff_eps

def drift_detector(diff_eps, window, threshold):

    #adwin = drift.ADWIN(delta = 0.2, clock = 32, max_buckets=5, min_window_length=5, grace_period=10)
    adwin = drift.KSWIN()
    for i, val in enumerate(diff_eps):

        adwin.update(val)

        if adwin.drift_detected:

            print(i+4000, "change detected")







    # for i in range(100, len(diff_eps), window):

    #     mean, std = np.mean(diff_eps[i:i+window]), np.std(diff_eps[i:i+window])

    #     if mean> ref_mean * threshold:
    #         print("instance ", i+4000)
    #         print("refmean ", ref_mean, "mean ", mean)

        
    #     if std > ref_std* threshold:
    #         print("instance ", i+4000)
    #         print("refstd ", ref_std, "std ", std)
    
###########################################

epsilon_sep = "results/SEA_s_10_l.csv"
epsilon_dec = "results/SEA_s_10_l_pred.csv"
threshold = 20
window = 100

#retrieve seperator epsilon values and decision boundary epsilon values
sep_epsilons, dec_epsilons = read_csv(epsilon_sep, epsilon_dec)

#extract epsilon values
diff_eps = extract(sep_epsilons, dec_epsilons)

#detect drift
drift_detector(diff_eps, window, threshold)

