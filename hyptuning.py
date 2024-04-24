from skorch import NeuralNetClassifier
import random
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from skorch.dataset import ValidSplit
import os
import time
from scipy.stats import randint, loguniform


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = str(seed)

class Net_3_2(nn.Module):
    def __init__(self):
        super(Net_3_2, self).__init__()
        self.fc1 = nn.Linear(3,2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output


def train_pipeline():

    #set random seed for reproducability


    dataset_name = "SEA_0_1"


    #retrieve features and labels
    features = np.load(f"datasets/features_{dataset_name}.npy")
    labels = np.load(f"datasets/labels_{dataset_name}.npy")

    #only first 2000 instances

    features = features[:2000]
    labels = labels[:2000]

    #turn them in pytorch format
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).long()

    model = NeuralNetClassifier(
        module = Net_3_2,
        criterion = nn.CrossEntropyLoss,
        verbose = False
    )

    # define the grid search parameters
    param_grid = {
        'batch_size': [16, 32, 64, 128],
        'max_epochs': [5, 10, 25, 50, 100],
        'optimizer__lr': [0.01, 0.005, 0.001, 0.0005, 0.0001],
        'optimizer': [optim.SGD, optim.RMSprop, optim.Adam],
    }
    # param_grid = {
    #     'batch_size' : randint(5,129),
    #     'max_epochs' : randint(5,101),
    #     'optimizer__lr' : loguniform(0.0001, 0.1),
    #     'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad,
    #                optim.Adam]
    # }
    start = time.time()
    #grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=60, n_jobs=1, cv=5)
    grid = GridSearchCV(estimator=model, param_grid= param_grid, n_jobs=1, cv=5)
    grid_result = grid.fit(features, labels)
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("It took", length, "seconds!")

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("It took", length, "seconds!")


train_pipeline()


