import random
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

#TRAINING PIPELINE
class Train():
    def __init__(self, dataset_name, training_instances, shuffle, model, 
                 batch_size, epochs, learning_rate, visualize):
        self.dataset_name = dataset_name
        self.training_instances = training_instances
        self.shuffle = shuffle
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.visualize = visualize

    def retrieve_dataset(self)-> torch:
        #retrieve features and labels
        features = np.load(f"datasets/features_{self.dataset_name}.npy")
        labels = np.load(f"datasets/labels_{self.dataset_name}.npy")

        #turn them in pytorch format
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()

        #prepare dataloader
        train_dataset = TensorDataset(features[0:self.training_instances], labels[0:self.training_instances])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return features, labels, train_loader
    
    def retrieve_model(self)-> torch:

        if self.model == "Net_3_2":
            model = Net_3_2()
        else:
            print(f"Model {self.model} does not yet exist.")

        return model
    
    def train_model(self, train_loader, model):
        #set criterion and optimizer
        criterion = nn.CrossEntropyLoss() #binary cross entropy loss
        optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)

        for epoch in range(self.epochs):
            total_correct, total_samples = 0, 0

            for batch_X, batch_y in train_loader:
                #forward pass
                x = model(batch_X)

                #compute loss
                loss = criterion(x, batch_y)

                #backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #compute some statistics
                with torch.no_grad():
                    #print(torch.softmax(x, dim=1))
                    predicted = torch.argmax(torch.softmax(x, dim=1), dim=1)
                    #predicted = (x.values>0.5).long()
                    correct_predictions = (predicted==batch_y).sum().item() #torch.argmax(batch_y, dim=1)


                    total_correct+= correct_predictions
                    total_samples += len(batch_y)

            print("loss: ", loss, " for epoch: ", epoch)
            print("accuracy: ", total_correct/total_samples)
            print(total_samples)
        
        #save model
        torch.save(model.state_dict(), f"model_weights/{self.dataset_name}.pth")

    def visualize_drift(self, acc: list):
        instances = [i for i in range(len(acc))]

        plt.plot(instances, acc, marker = "o", label = "nn")

        plt.xlabel("#instances x 100")
        plt.ylabel("accuracy")
        plt.tight_layout()
        plt.legend()
        plt.show()

    def visualize_model(self, features: torch, labels: torch, model: torch):

        #load model weights
        model.load_state_dict(torch.load(f"model_weights/{self.dataset_name}.pth"))
        
        #set results:
        results = []

        for i in range(0,len(labels), 100):
            print(i, "current index")

            #forward pass only
            with torch.no_grad():
                x = model(features[i:i+100])

                #get predictions
                predicted = torch.argmax(torch.softmax(x, dim=1), dim=1)

                #get correct predictions
                correct_predictions = (predicted==labels[i:i+100]).sum().item()

            #save some statistics    
            results.append(correct_predictions/100)
            print("accuracy: ", correct_predictions/100)

        #visualize the drift over all instances (train and test)
        self.visualize_drift(results)

    def get_predictions(self, features, labels, model):

        pred_labels = []

        #load model weights
        model.load_state_dict(torch.load(f"model_weights/{self.dataset_name}.pth"))

        for i in range(0,len(labels)):
            print(i, "this is the current index")

            #forward pass only
            with torch.no_grad():
                x = model(features[i])
                predicted = torch.argmax(torch.softmax(x, dim=0), dim=0)
                pred_labels.append(predicted.detach().cpu())
        
        #save predicted labels
        np.save(f"datasets/labels_{self.dataset_name}_pred.npy", pred_labels)

    def train_pipeline(self):

        #set random seed for reproducability
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        #prep training instances
        features, labels, train_loader = self.retrieve_dataset()

        #retrieve correct model
        model = self.retrieve_model()

        #actual train pipeline
        self.train_model(train_loader, model)

        #visualize model if desired
        if self.visualize:
            self.visualize_model(features, labels, model)
        
        #get predictions based on trained model
        self.get_predictions(features, labels, model)
        




        
#ALL NETS USED FOR EXPERIMENTS
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
    




