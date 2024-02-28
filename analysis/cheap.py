import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim

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

def prep_data(feature_name, label_name, epsilons_act, epsilons_pred, model_name, begin_idx, end_idx):
    #retrieve epsilon values
    epsilons_act = pd.read_csv(epsilons_act)
    epsilons_act = list(epsilons_act["epsilon"])

    epsilons_pred = pd.read_csv(epsilons_pred)
    epsilons_pred = list(epsilons_pred["epsilon"])

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

    return features_torch, labels_torch, model, epsilons_act, epsilons_pred


data_name = "sd_03"
begin_idx = 2000
end_idx = 8000
split = 5000


#retrieve features + labels
feature_name = f"datasets_42/SEA_features_{data_name}.npy"
label_name = f"datasets_42/SEA_labels_{data_name}.npy"
#retrieve epsilon values
epsilon_act = f"results/SEA_{data_name}.csv"
epsilon_pred = f"results/SEA_{data_name}_pred.csv"
#retrieve trained static model
model_name = f"model_weights_42/SEA_staticnn_{data_name}.pth"

features_torch, labels_torch, model, epsilons_act, epsilons_pred = prep_data(feature_name, label_name, epsilon_act, epsilon_pred, model_name, begin_idx, end_idx)

