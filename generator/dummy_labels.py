import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3,2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output

def prep_data(feature_name, model_name):

    #retrieve features and labels
    features = np.load(feature_name)

    #torch features
    features_torch = torch.from_numpy(features).float()
    
    #retrieve model
    model = Net()
    model.load_state_dict(torch.load(model_name))

    return features_torch, model


def get_labels(features, model):
    labels = []

    for i in range(0, 10000):
        print(i, "dit is i")

        #forward pass
        with torch.no_grad():
            x = model(features[i])
            predicted = torch.argmax(torch.softmax(x, dim=0), dim=0)
            labels.append(predicted.detach().cpu())
    print(labels[3027])
    np.save("SEA_labels_s_12_23_dummy.npy", labels)

data_name = "s_12_23"

#retrieve features + labels
feature_name = f"datasets_42/SEA_features_{data_name}.npy"
#retrieve trained static model
model_name = f"model_weights_42/SEA_staticnn_{data_name}.pth"

features_torch, model = prep_data(feature_name, model_name)
get_labels(features_torch, model)

                