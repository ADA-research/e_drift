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
from scipy import stats

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


def cdf(ref_win, test_win):


    binsa, binsb = [], []
    cdfs = []

    fig1, ax1 = plt.subplots()
    #getting data of the histogram
    counta, bins_counta = np.histogram(ref_win, bins = 100)
    countb, bins_countb = np.histogram(test_win, bins = 100)
    
    binsa.append(bins_counta)
    binsb.append(bins_countb)

    pdfa = counta/sum(counta)
    pdfb = countb/sum(countb)

    cdfa = np.cumsum(pdfa)
    cdfb = np.cumsum(pdfb)

    ax1.plot(bins_counta[1:], cdfa, label = f"ref window")
    ax1.plot(bins_countb[1:], cdfb, label = f"test window")

    #plt.plot(bins[0][1:], cdfs, color = "red", label = "reference window (4800-5000)")
    plt.xlabel("lower bound to critical epsilon")
    plt.legend()
    plt.show()


def drift_detector(diff_eps, window, threshold):

    ref_win = diff_eps[0:499]
    test_win_values = []

    count = 0
    
    for i in range(499,len(diff_eps)):

        if len(test_win_values)==500:

            res = stats.ks_2samp(test_win_values, ref_win)
            if res[1]< 0.0001:

                if i+3000>5000:
                    cdf(ref_win, test_win_values)
                    print("statistical significance found at ", i+3000)
                    break
                else:
                    count+=1
                    print("FPS: ", count)
            test_win_values.pop(0)
            test_win_values.append(diff_eps[i])

        else:

            test_win_values.append(diff_eps[i])

        

        
    
###########################################

epsilon_sep = "results/HYP_m001.csv"
epsilon_dec = "results/HYP_m001_pred.csv"
threshold = 100
window = 100

#retrieve seperator epsilon values and decision boundary epsilon values
sep_epsilons, dec_epsilons = read_csv(epsilon_sep, epsilon_dec)

#extract epsilon values
diff_eps = extract(sep_epsilons, dec_epsilons)

#detect drift
drift_detector(diff_eps, window, threshold)

