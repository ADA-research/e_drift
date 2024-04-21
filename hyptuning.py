from skorch import NeuralNetClassifier
import random
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

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

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    model = NeuralNetClassifier(
        module = Net_3_2,
        criterion = nn.CrossEntropyLoss,
        optimizer = optim.Adam,
        verbose = False
    )

    # define the grid search parameters
    param_grid = {
        'batch_size': [10, 20, 30],
        'max_epochs': [10, 50, 100]
    }


    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
    grid_result = grid.fit(features, labels)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    


train_pipeline()


