import yaml
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils import write_yaml_HPO


def get_dataset():

    features = np.load(f"datasets/features_electricity.npy")
    labels = np.load(f"datasets/labels_electricity.npy")
    predictions = np.load(f"datasets/labels_electricity_pred.npy")

    return features, labels, predictions

def no_change():

    pass

def majority_class():

    pass

def neural_network():

    pass

def visualize():
    
    pass

def main():

    #read in original features, labels and predicted labels
    features, labels, predictions = get_dataset()
    

if __name__ == '__main__':
    main()