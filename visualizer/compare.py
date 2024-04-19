import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random
from scipy import stats
import matplotlib.patches as mpatches
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


def drift_detector(diff_eps,begin_idx, confidence):

    ref_win = diff_eps[0:999]
    test_win_values = []

    fp_count = 0
    tp_pos = None
    
    for i in range(1000,len(diff_eps)):

        if len(test_win_values)==1000:

            res = stats.mannwhitneyu(test_win_values, ref_win)
            if res[1]< confidence:
                print(res[1])
                print(res)

                if i+begin_idx>5000:
                    #cdf(ref_win, test_win_values)
                    print("statistical significance found at ", i+begin_idx)
                    tp_pos = i+begin_idx - 5000
                    break
                else:
                    fp_count+=1
                    print("FPS: ", fp_count)
            test_win_values.pop(0)
            test_win_values.append(diff_eps[i])

        else:

            test_win_values.append(diff_eps[i])
    
    return fp_count, tp_pos 


def drift_detector1(diff_eps,begin_idx, confidence):

    ref_win = diff_eps[0:999]
    test_win_values = []

    fp_count = 0
    tp_pos = None
    
    for i in range(1000,len(diff_eps)):

        if len(test_win_values)==1000:

            res = stats.ks_2samp(test_win_values, ref_win)
            if res[1]< confidence:
                print(res[1])
                print(res)

                if i+begin_idx>5000:
                    #cdf(ref_win, test_win_values)
                    print("statistical significance found at ", i+begin_idx)
                    tp_pos = i+begin_idx - 5000
                    break
                else:
                    fp_count+=1
                    print("FPS: ", fp_count)
            test_win_values.pop(0)
            test_win_values.append(diff_eps[i])

        else:

            test_win_values.append(diff_eps[i])
    
    return fp_count, tp_pos 

def drift_detector2(diff_eps,begin_idx, confidence):
    fp_count = 0
    tp_pos = None

    kswin = drift.KSWIN(alpha=confidence, seed=42)

    for i, val in enumerate(diff_eps):

        kswin.update(val)

        if kswin.drift_detected:

            if i+begin_idx>5000:
                print("statistical significance found at ", i+begin_idx)
                tp_pos = i+begin_idx - 5000
                break
            else:
                fp_count+=1
    print(fp_count, "kswin fp count")
    return fp_count, tp_pos 

def drift_detector3(diff_eps,begin_idx, confidence):
    fp_count = 0
    tp_pos = None

    adwin = drift.ADWIN(delta=confidence)

    for i, val in enumerate(diff_eps):

        adwin.update(val)

        if adwin.drift_detected:

            if i+begin_idx>5000:
                print("statistical significance found at ", i+begin_idx)
                tp_pos = i+begin_idx - 5000
                break
            else:
                fp_count+=1
    print(fp_count, "adwin fp count")
    return fp_count, tp_pos 

def drift_detector4(diff_eps,begin_idx, confidence):
    fp_count = 0
    tp_pos = None

    adwin = drift.PageHinkley(delta=confidence)

    for i, val in enumerate(diff_eps):

        adwin.update(val)

        if adwin.drift_detected:

            if i+begin_idx>5000:
                print("statistical significance found at ", i+begin_idx)
                tp_pos = i+begin_idx - 5000
                break
            else:
                fp_count+=1
    print(fp_count, "ph fp count")
    return fp_count, tp_pos 


def graph(fp_counts, tp_positions, confidences):
    names = ["mannwhitneyu","kolmogorov smirnov test", "KSWIN", "ADWIN", "PageHinkley"]

    fig1, ax1 = plt.subplots()

    for i in range(len(names)):

        ax1.plot(tp_positions[i], fp_counts[i], "-o", label = names[i])

        for j in range(len(tp_positions[i])):
            ax1.annotate(confidences[j], (tp_positions[i][j], fp_counts[i][j]+0.001), ha='center', rotation=60)

def graph1(tp_positions, fp_counts, confidences):
    plt.rcParams["figure.figsize"] = (5,5)
    names = ["mannwhitneyu","kolmogorov smirnov test", "KSWIN", "ADWIN", "PageHinkley"]
    fig1, ax1 = plt.subplots()
    for i in range(len(names)):
        new_tp, new_fp = [], []
        # for tp, fp in zip(tp_positions[i], fp_counts[i]):
        #     if tp <1500:
        #         new_tp.append(tp)
        #         new_fp.append(fp)
        # tp_positions[i] = new_tp
        # fp_counts[i] = new_fp
        x_points = np.empty(len(tp_positions[i]))
        x_points.fill(i)
        color_list = []
        for fp in fp_counts[i]:

            if fp==0:
                color_list.append("green")

            elif fp <5:
                color_list.append("yellow")
            
            elif fp <10:
                color_list.append("orange")
            
            else:
                color_list.append("red")

        ax1.scatter(x_points, tp_positions[i], c=color_list, cmap="viridis")
        ax1.plot(x_points, tp_positions[i], linestyle = "-", color="black")


        for j in range(len(tp_positions[i])):
            #ax1.plot(x_points[j], tp_positions[i][j], marker = "o", c=color_list[j])
            ax1.annotate(confidences[j], (x_points[j],tp_positions[i][j]), ha='left', rotation=60)
    green_patch = mpatches.Patch(color='green', label='FPS = 0')
    yellow_patch = mpatches.Patch(color='yellow', label='FPS < 5')
    orange_patch = mpatches.Patch(color='orange', label='FPS < 10')
    red_patch = mpatches.Patch(color='red', label='FPS >= 10')
    x_ticks = np.arange(0,5)
    plt.xticks(x_ticks, names)
    plt.xlabel("detection mechanism")
    plt.ylabel("TP detection position")
    #plt.ylim(-0.1, 1500)
    plt.title("HYP m=0.001")
    plt.yscale("log")
    plt.margins(0.2)
    plt.legend(handles=[green_patch, yellow_patch, orange_patch, red_patch])
    #plt.colorbar()
    plt.show()



epsilon_sep = "results/HYP_m001.csv"
epsilon_dec = "results/HYP_m001_pred.csv"
begin_idx = 3000

#retrieve seperator epsilon values and decision boundary epsilon values
sep_epsilons, dec_epsilons = read_csv(epsilon_sep, epsilon_dec)

#extract epsilon values
diff_eps = extract(sep_epsilons, dec_epsilons)


confidences = [0.05, 0.01, 0.005, 0.001, 0.0001]
fp_counts, tp_positions = [], []
fp_count_mann, tp_pos_mann =[], []
fp_count_ks, tp_pos_ks = [], []
fp_count_kswin, tp_pos_kswin = [], []
fp_count_adwin, tp_pos_adwin = [], []
fp_count_ph, tp_pos_ph = [], []

for confidence in confidences:
    #detect drift
    fp_count, tp_pos = drift_detector(diff_eps, begin_idx, confidence)
    if tp_pos !=None:
        fp_count_mann.append(fp_count)
        tp_pos_mann.append(tp_pos)
    fp_count, tp_pos = drift_detector1(diff_eps, begin_idx, confidence)
    if tp_pos !=None:
        fp_count_ks.append(fp_count)
        tp_pos_ks.append(tp_pos)
    fp_count, tp_pos = drift_detector2(diff_eps, begin_idx, confidence)
    if tp_pos !=None:
        fp_count_kswin.append(fp_count)
        tp_pos_kswin.append(tp_pos)

    fp_count, tp_pos = drift_detector3(diff_eps, begin_idx, confidence)
    if tp_pos !=None:
        fp_count_adwin.append(fp_count)
        tp_pos_adwin.append(tp_pos)

    fp_count, tp_pos = drift_detector4(diff_eps, begin_idx, confidence)
    if tp_pos !=None:
        fp_count_ph.append(fp_count)
        tp_pos_ph.append(tp_pos)

fp_counts.append(fp_count_mann)
fp_counts.append(fp_count_ks)
fp_counts.append(fp_count_kswin)
fp_counts.append(fp_count_adwin)
fp_counts.append(fp_count_ph)

tp_positions.append(tp_pos_mann)
tp_positions.append(tp_pos_ks)
tp_positions.append(tp_pos_kswin)
tp_positions.append(tp_pos_adwin)
tp_positions.append(tp_pos_ph)
#generate graph

#graph(fp_counts, tp_positions, confidences)
graph1(tp_positions, fp_counts, confidences)
