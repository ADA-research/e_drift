import random
import torch
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from utils import write_yaml_HPO

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

        
        return features, labels
    
    def retrieve_model(self)-> torch:

        if self.model == "Net_3_2":
            model = Net_3_2()
        else:
            print(f"Model {self.model} does not yet exist.")

        return model
    
    def HPO(self, features, labels, network):

        #get only training features and labels
        features = features[0:self.training_instances]
        labels = labels[0:self.training_instances]

        #define the grid search parameters
        param_grid = {
            'batch_size': [16, 32, 64, 128],
            'max_epochs': [5, 10, 25, 50, 100],
            'optimizer__lr': [0.01, 0.005, 0.001, 0.0005, 0.0001],
            'optimizer': [optim.SGD, optim.RMSprop, optim.Adam],
        }

        #set skorch model
        model = NeuralNetClassifier(
            module = network,
            criterion = nn.CrossEntropyLoss,
            verbose = False
        )

        #perform grid search with 5-fold cross-validation
        grid = GridSearchCV(estimator=model, param_grid= param_grid, n_jobs=-1, scoring = make_scorer(matthews_corrcoef), cv=5)
        grid_result = grid.fit(features, labels)

        #return best params
        print(grid_result.best_params_)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

            
        write_yaml_HPO("main_config.yaml", grid_result.best_params_)
        return grid_result.best_params_
    
    def train_model(self, features, labels, model, best_params):
        #prepare dataloader
        train_dataset = TensorDataset(features[0:self.training_instances], labels[0:self.training_instances])
        train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=self.shuffle)

        #set criterion and optimizer
        criterion = nn.CrossEntropyLoss() #binary cross entropy loss
        #optimizer = optim.Adam(model.parameters(), lr = self.learning_rate)
        optimizer = best_params["optimizer"](model.parameters(), lr = best_params["optimizer__lr"])

        #for epoch in range(self.epochs):
        for epoch in range(best_params["max_epochs"]):
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

    def visualize_model(self, features, labels, model):

        #load model weights
        model.load_state_dict(torch.load(f"model_weights/{self.dataset_name}.pth"))
        
        #set results:
        results = []

        for i in range(0,10000, 100):
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
        features, labels = self.retrieve_dataset()

        #retrieve correct model
        model = self.retrieve_model()

        #perform HPO
        best_params = self.HPO(features, labels, model)

        #actual train pipeline
        self.train_model(features, labels, model, best_params)

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
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(2,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        output = self.fc2(x)
        return output
    




