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

    print(len(diff_eps), "wtf")

    return diff_eps

def track_diff(diff_eps):


    max_dif = -1

    max_dif = np.max(diff_eps[0:1000])

    print(diff_eps[1000:2000])

    for idx, eps in enumerate(diff_eps):
        if eps>max_dif:
            print("eps: ", eps, "index: ", idx+4000)
    
    print(np.mean(diff_eps[0:1000]), np.std(diff_eps[0:1000]))
    print(np.mean(diff_eps[1000:2000]), np.std(diff_eps[1000:2000]))


def kswin(diff_eps, window_size, stat_size, alpha):

    kswin = drift.KSWIN(window_size= window_size, stat_size = stat_size, alpha = alpha, seed=42)


    for idx, eps in enumerate(diff_eps):

        kswin.update(eps)

        if kswin.drift_detected:

            print("index: ", idx+4000, eps)



def plot(diff_eps):
    x_values = np.arange(0,len(diff_eps))
    plt.plot(x_values, diff_eps, marker='o', linestyle='-')
    plt.xlabel('time')
    plt.ylabel('critical epsilon value')
    plt.show()

def violin_plot(diff_eps, window, begin_idx, start, end):

    df = pd.DataFrame(columns = ["time","lower bound to critical epsilon"])

    for i in range(start, end, window):

        eps_vals = diff_eps[i:i+window]
        print(eps_vals, "values")
        for idx, eps in enumerate(eps_vals):
            if eps>0:
                df.loc[len(df.index)] = [i+window+begin_idx, eps]
    
    sns.boxplot(data=df, x="time", y="lower bound to critical epsilon")
    plt.axvline(x=1.5, color = "red")
    plt.legend(loc='upper right')
    plt.show()




epsilon_sep = "results/SEA_s_100_f.csv"
epsilon_dec = "results/SEA_s_100_f_pred.csv"


window_size = 300
stat_size = 100
alpha = 0.01


start = 0
end = 6000
window = 100
begin_idx = 0

#read in csv files
sep_epsilons, dec_epsilons = read_csv(epsilon_sep, epsilon_dec)

#extract epsilon values
diff_eps = extract(sep_epsilons, dec_epsilons)


#track_diff(diff_eps)

#kswin(diff_eps, window_size, stat_size, alpha)
plot(diff_eps)

violin_plot(diff_eps, window, begin_idx, start, end)
# TP, FP, earlier_det = drift_detection(diff_eps, alpha, window_size, stat_size, seed, begin_idx, threshold)

# print(TP)
# print(FP)
# print(earlier_det)

# print("number of True positives: ", len([1 for i in TP if i >0]))
# print("number of False positives: ", np.sum([i for i in FP if i >0]), np.sum(FP))
# print("average detection position: ", np.mean([i for i in earlier_det if i>0]))
