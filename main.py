import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import onnx
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#set random seed for reproducability
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100,64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64,32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32,16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16,8)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(8,4)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(4,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        output = self.fc6(x)
        return output
    
def train_network(features, labels, model_name):

    # model = nn.Sequential(
    #     nn.Linear(3, 8),
    #     nn.ReLU(),
    #     nn.Linear(8,2)
    # )
    model = Net()
    criterion = nn.CrossEntropyLoss() #binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).long()  #was long

    train_dataset = TensorDataset(features[0:2000], labels[0:2000])

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) #shuffle = True gives better results than shuffle = False, also reproducability is affected
    

    num_epochs = 25
    for epoch in range(num_epochs):
        total_correct = 0
        total_samples = 0
        for batch_X, batch_y in train_loader:
            #forward pass
            x = model(batch_X)
            #print(batch_y, "batch y")
            #retrieve highest logits
            #x = torch.max(x, dim=1, keepdim=True)
            #outputs = torch.sigmoid(x.values)

            #compute loss
            loss = criterion(x, batch_y)

            #backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    torch.save(model.state_dict(), model_name)

    #save to onxx
    # model.train(False)
    # dummy_input = torch.randn(1,3, requires_grad=True)
    # input_names = ["actual_input"]
    # output_names = ["output"]
    # torch.onnx.export(model, dummy_input, "staticnn.onnx", verbose=False, input_names = input_names, output_names = output_names, export_params= True)

def test_network(features, labels, model_name):

    results = []

    #load model
    model = Net()
    model.load_state_dict(torch.load(model_name))

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()


    for i in range(0, 10000, 100):
        print(i, "dit is i")

        #forward pass
        with torch.no_grad():
            x = model(features[i:i+100])
            #x = model(torch.FloatTensor([[0.58641319, 0.85209298, 0.29021813], [0.58641319-0.323, 0.85209298-0.323, 0.29021813-0.323]]))
            #print(x)
        
        #print(prediction)
            predicted = torch.argmax(torch.softmax(x, dim=1), dim=1)
            #print(predicted)
            #dfn()
            #print(torch.softmax(x, dim=1))
            #predicted = (x.values>0.5).long()
            correct_predictions = (predicted==labels[i:i+100]).sum().item()

        results.append(correct_predictions/100)
        print("accuracy: ", correct_predictions/100)
        
    return results

def visualize_drift(acc):
    instances = [i for i in range(len(acc))]

    plt.plot(instances, acc, marker = "o", label = "nn")

    plt.xlabel("#instances x 100")
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

def get_dataset(feature_name, label_name):
    new_labels = []
    features = np.load(feature_name)
    labels = np.load(label_name)

    # print(features)
    # print(scaler.mean_)
    # print(np.sqrt(scaler.var_))
    # print((10 - scaler.mean_) / np.sqrt(scaler.var_))
    # print((0 - scaler.mean_) / np.sqrt(scaler.var_))
    
    #DIT WERKT HIERONDER WERKT NIET VOOR ALPHABETACROWN
    # for lab in labels: 
    #     if lab == True:
    #         new_labels.append([0.,1.])
    #     else:
    #         new_labels.append([1.,0.])
    # #labels = np.expand_dims(labels, axis=1)
    # new_labels = np.array(new_labels)
    return features, labels


def load_onnx():
    model = onnx.load("staticnn.onnx")
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

def main():
    feature_name = "datasets/SEA_features_s_100_f.npy"
    label_name = "datasets/SEA_labels_s_100_f.npy"
    model_name = "model_weights/SEA_staticnn_s_100_f.pth"

    features, labels = get_dataset(feature_name, label_name)
    train_network(features, labels, model_name)
    #load_onnx()
    acc = test_network(features, labels, model_name)
    visualize_drift(acc)

if __name__ == '__main__':
    main()