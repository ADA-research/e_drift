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

    #get mean and std of training data (first 1000 instances)

    ref_mean = np.mean(diff_eps[0:500])
    ref_std = np.std(diff_eps[0:500])

    print("refmean ", ref_mean, "refstd ", ref_std)

    ##sliding window approach

    mean_window = []
    std_window = []

    count_m = 0
    count_s = 0

    for i in range(500, len(diff_eps)):

        if len(mean_window) == window:
            if np.mean(mean_window)> ref_mean * threshold:
                count_m +=1

                print("instance ", i+2000)
                print("refmean ", ref_mean, "mean ", np.mean(mean_window))
                if i+2000>5000:
                    print(count_m)

                    break
            
            mean_window.pop(0)
            mean_window.append(diff_eps[i])

        
        else:
            mean_window.append(diff_eps[i])


        if len(std_window) == window:
            if np.std(std_window) > ref_std* threshold:
                count_s +=1
                print("instance ", i+2000)
                print("refstd ", ref_std, "std ", np.std(std_window))
                if i+2000>5000:
                    print(count_s)

                    break
            std_window.pop(0)
            std_window.append(diff_eps[i])

        else:
            std_window.append(diff_eps[i])





    # for i in range(100, len(diff_eps), window):

    #     mean, std = np.mean(diff_eps[i:i+window]), np.std(diff_eps[i:i+window])

    #     if mean> ref_mean * threshold:
    #         print("instance ", i+4000)
    #         print("refmean ", ref_mean, "mean ", mean)

        
    #     if std > ref_std* threshold:
    #         print("instance ", i+4000)
    #         print("refstd ", ref_std, "std ", std)
    
###########################################

epsilon_sep = "results/SEA_s_12_3.csv"
epsilon_dec = "results/SEA_s_12_3_pred.csv"
threshold = 100
window = 100

#retrieve seperator epsilon values and decision boundary epsilon values
sep_epsilons, dec_epsilons = read_csv(epsilon_sep, epsilon_dec)

#extract epsilon values
diff_eps = extract(sep_epsilons, dec_epsilons)

#detect drift
drift_detector(diff_eps, window, threshold)

