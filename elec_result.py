import yaml
import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from river import drift

class Net_7_2(nn.Module):
    def __init__(self):
        super(Net_7_2, self).__init__()
        self.fc1 = nn.Linear(7,4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4,2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(2,2)

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.fc3(x)
        return output

def get_dataset():

    features = np.load(f"datasets/features_electricity.npy")
    labels = np.load(f"datasets/labels_electricity.npy")
    predictions = np.load(f"datasets/labels_electricity_pred.npy")

    return features, labels, predictions

def no_change(labels):

    predictions = labels[1999:len(labels)-1]
    mcc = matthews_corrcoef(labels[2000:], predictions)

    return predictions, mcc

def majority_class(labels):

    predictions = []
    moving_window = list(labels[1000:2000])
    
    for i in range(2000, len(labels)):
        counts = np.bincount(moving_window)
        predictions.append(np.argmax(counts)) 

        moving_window.pop(0)
        moving_window.append(labels[i])

    mcc = matthews_corrcoef(labels[2000:], predictions)
    return predictions, mcc

def neural_network(labels, predictions):

    mcc = matthews_corrcoef(labels[2000:], predictions[2000:])

    return predictions[2000:], mcc

def retrain_network(retraining_features, retraining_labels, model):
    #convert numpy arrays to pytorch 
    features = torch.tensor(np.array(retraining_features)).float()
    labels = torch.tensor(np.array(retraining_labels, dtype=int)).long()

    #prepare dataloader
    train_dataset = TensorDataset(features, labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    #set criterion and optimizer
    criterion = nn.CrossEntropyLoss() #binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    for epoch in range(100):

        for batch_X, batch_y in train_loader:
            #forward pass
            x = model(batch_X)

            #compute loss
            loss = criterion(x, batch_y)

            #backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    #save model
    torch.save(model.state_dict(), f"data_el/electricity.pth")

    
def error_rate(features, labels):

    drift_counter = 0
    predictions = []

    #set retraining windows (features + labels)
    retraining_features, retraining_labels = list(features[1000:2000]), list(labels[1000:2000])

    #convert numpy arrays to pytorch 
    features = torch.tensor(features).float()
    labels = torch.tensor(labels).long()



    #load network
    model = Net_7_2()
    model.load_state_dict(torch.load(f"data_el/electricity.pth"))

    #initialze DDM
    ddm = drift.binary.DDM()

    for idx, feature in enumerate(features[2000:]):
        idx+=2000
        print("idx and number of retrainings: ", idx, drift_counter)

        #forward pass
        with torch.no_grad():
            x = model(feature)
            predicted = torch.argmax(torch.softmax(x, dim=0), dim=0)
            predicted = predicted.detach().cpu()
        
        #save predictions
        predictions.append(predicted)

        #update drift detector
        ddm_input = int(not predicted == labels[idx])
        ddm.update(ddm_input)

        #update retraining windows
        retraining_features.pop(0)
        retraining_features.append(feature.tolist())

        retraining_labels.pop(0)
        retraining_labels.append(labels[idx].tolist())
        

        #check if drift is detected
        if ddm.drift_detected:
            drift_counter+=1

            #retrain network
            retrain_network(retraining_features, retraining_labels, model)

            #reload model weights
            model = Net_7_2()
            model.load_state_dict(torch.load(f"data_el/electricity.pth"))

            #reset drift detector
            ddm = drift.binary.DDM()

    mcc = matthews_corrcoef(labels[2000:], predictions)
    print(mcc)
    print(drift_counter)

    return predictions, mcc, drift_counter


def visualize():
    
    pass

def main():

    #read in original features, labels and predicted labels
    features, labels, predictions = get_dataset()

    # #no-change classifier
    # predictions_no_change, mcc_no_change = no_change(labels)
    # print("mcc no-change classifier: ", mcc_no_change)

    # #majority class classifier 
    # predictions_majority_class, mcc_majority_class = majority_class(labels)
    # print("mcc majority-class classifier: ", mcc_majority_class)

    # #neural network without retraining
    # predictions_neural_network, mcc_neural_network = neural_network(labels, predictions)
    # print("mcc neural network: ", mcc_neural_network)

    #error-rate with DDM
    predictions, mcc, drift_counter = error_rate(features, labels)
    



    

if __name__ == '__main__':
    main()