import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


#network for 10 features
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8,4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        output = self.fc3(x)
        return output

def prep_data(feature_name, label_name, epsilon_name, model_name, begin_idx, end_idx):
    #retrieve epsilon values
    epsilon_values = pd.read_csv(epsilon_name)
    epsilon_values = list(epsilon_values["epsilon"])

    #retrieve features and labels
    features = np.load(feature_name)
    features = features[begin_idx:end_idx]
    labels = np.load(label_name)
    labels = labels[begin_idx:end_idx]

    #torch features and labels
    features_torch = torch.from_numpy(features).float()
    labels_torch = torch.from_numpy(labels).float()
    
    #retrieve model
    model = Net()
    model.load_state_dict(torch.load(model_name))

    return features_torch, labels_torch, epsilon_values, model

def signal_drift(features, labels, epsilon_values, model):

        for i in range(1000, 1250):
            print(i+4000, "dit is i")

            #forward pass
            with torch.no_grad():
                x = model(features[i])
                predicted = torch.argmax(torch.softmax(x, dim=0), dim=0)
                correct_prediction = (predicted==labels[i])

                
                if not correct_prediction:
                    print("label: ", labels[i], "predicted: ", predicted)
                    print(features[i])
                    print(epsilon_values[i])
                 


data_name = "s_10_l"
begin_idx = 4000
end_idx = 6000
split = 5000

#retrieve features + labels
feature_name = f"datasets/SEA_features_{data_name}.npy"
label_name = f"datasets/SEA_labels_{data_name}.npy"
#retrieve epsilon values
epsilon_name = f"results/SEA_{data_name}.csv"
#retrieve trained static model
model_name = f"model_weights/SEA_staticnn_{data_name}.pth"

features_torch, labels_torch, epsilon_values, model = prep_data(feature_name, label_name, epsilon_name, model_name, begin_idx, end_idx)
signal_drift(features_torch, labels_torch, epsilon_values, model)