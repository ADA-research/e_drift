import pandas as pd
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def prep_data(feature_name, label_name_act, label_name_pred, epsilon_act, epsilon_pred, model_name, begin_idx, end_idx):


    epsilon_values_act = pd.read_csv(epsilon_act)
    epsilon_values_act = list(epsilon_values_act["epsilon"])

    epsilon_values_pred = pd.read_csv(epsilon_pred)
    epsilon_values_pred = list(epsilon_values_pred["epsilon"])

    #retrieve features and labels
    features = np.load(feature_name)
    features = features[begin_idx:end_idx]
    labels_act = np.load(label_name_act)
    labels_act = labels_act[begin_idx:end_idx]
    labels_pred = np.load(label_name_pred)
    labels_pred = labels_pred[begin_idx:end_idx]


    return features, labels_act, labels_pred, epsilon_values_act, epsilon_values_pred


def generate_csv(features, labels_act, labels_pred, epsilon_values_act, epsilon_values_pred):

    data = []

    for i in range(len(labels_act)):

        instance = []

        






eps_actual = "s_10_l"
eps_pred = "s_10_l_pred"
begin_idx = 4000
end_idx = 6000
split = 5000

#retrieve features + labels
feature_name = f"datasets/SEA_features_{eps_actual}.npy"
label_name_act = f"datasets/SEA_labels_{eps_actual}.npy"
label_name_pred = f"datasets/SEA_labels_{eps_pred}.npy"
#retrieve epsilon values
epsilon_act = f"results/SEA_{eps_actual}.csv"
epsilon_pred = f"results/SEA_{eps_pred}.csv"
#retrieve trained static model
model_name = f"model_weights/SEA_staticnn_{eps_actual}.pth"

features, labels_act, labels_pred, epsilon_values_act, epsilon_values_pred = prep_data(feature_name, label_name_act, label_name_pred, epsilon_act, epsilon_pred, model_name, begin_idx, end_idx)

generate_csv(features, labels_act, labels_pred, epsilon_values_act, epsilon_values_pred)