from river.datasets import synth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def generate_data(mag_change, noise):


    # dataset1 = synth.ConceptDriftStream(
    # stream=synth.SEA(seed=seed, variant=var_reg),
    # drift_stream=synth.SEA(seed=seed, variant=var_drift),
    # seed=seed, position=position, width=width
    # )
    dataset1 = synth.Hyperplane(seed=42, n_features = 10, mag_change = 0.0, noise_percentage= noise, sigma = 0.0, n_drift_features = 0)
    dataset2 = synth.Hyperplane(seed=42, n_features = 10, mag_change = mag_change, noise_percentage= noise, sigma = 0.0, n_drift_features = 2)
    return dataset1, dataset2
    
def save_data(dataset1, dataset2, feature_name, label_name):

    features, labels = [], []

    for x, y in dataset1.take(5000):
        features.append(list(x.values()))
        labels.append(y)

    for x, y in dataset2.take(5000):
        features.append(list(x.values()))
        labels.append(y)
    features = np.array(features)
    labels = np.array(labels)

    np.save(feature_name, features)
    np.save(label_name, labels)

feature_name = "datasets/HYP_features_m=001.npy"
label_name = "datasets/HYP_labels_m=001.npy"
mag_change = 0.001
noise = 0.0

dataset1, dataset2 = generate_data(mag_change, noise)
#compare_dist(dataset1, dataset2)
save_data(dataset1, dataset2, feature_name, label_name)