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
    counter=0

    for sep_eps, dec_eps in zip(sep_epsilons, dec_epsilons):
    
        diff_eps.append(abs(sep_eps-dec_eps))

    return diff_eps

def drift_detection(diff_eps, alpha, window_size, stat_size, seed, begin_idx, threshold):

    FP = 0
    TP = 0

    kswin = drift.KSWIN(alpha = alpha, window_size=window_size, stat_size=stat_size, seed = seed)
    kswin._reset()
    earlier_det = -1

    for idx, eps in enumerate(diff_eps):

        kswin.update(eps)
        if kswin.drift_detected:

            if idx+begin_idx < threshold:

                FP+=1
            
            else:
                TP+=1

                if earlier_det == -1:
                    earlier_det = idx+begin_idx
    
    return TP, FP, earlier_det

def track_diff(diff_eps):


    max_dif = -1

    max_dif = np.max(diff_eps[0:1000])

    print(max_dif)

    for idx, eps in enumerate(diff_eps[1000:6000]):
        if eps>max_dif:
            print(eps, idx+3000)

epsilon_sep = "results/SEA_sd_13.csv"
epsilon_dec = "results/SEA_sd_13_pred.csv"
alpha = 0.001
window_size = 700
stat_size = 300
seed = 4
begin_idx = 2000
threshold = 5000


#read in csv files
sep_epsilons, dec_epsilons = read_csv(epsilon_sep, epsilon_dec)

#extract epsilon values
diff_eps = extract(sep_epsilons, dec_epsilons)


track_diff(diff_eps)

# TP, FP, earlier_det = drift_detection(diff_eps, alpha, window_size, stat_size, seed, begin_idx, threshold)

# print(TP)
# print(FP)
# print(earlier_det)

# print("number of True positives: ", len([1 for i in TP if i >0]))
# print("number of False positives: ", np.sum([i for i in FP if i >0]), np.sum(FP))
# print("average detection position: ", np.mean([i for i in earlier_det if i>0]))
