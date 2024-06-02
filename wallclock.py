import pandas as pd
import numpy as np
from river import drift
from scipy import stats
import torch
from torch import nn
import time

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
    
class Net_10_2(nn.Module):
    def __init__(self):
        super(Net_10_2, self).__init__()
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



def determine_wallclock(dataset_names):

    min_time, max_time, mean_time = None,None,0
    all_times = []

    #loop over all datasets
    for dataset in dataset_names:
        print(dataset, "dataset")
        for i in range(1,6):
            data = pd.read_csv(f"results/{dataset}_{i}.csv")
            running_time = data["runtime"].to_list()
            running_time = [run_time for run_time in running_time if run_time>0]
            min_data = np.min(running_time)
            max_data = np.max(running_time)
            print(min_data, max_data)

            if min_time == None or min_data < min_time:
                min_time = min_data

            if max_time == None or max_data > max_time:
                max_time = max_data
            all_times.extend(running_time)
    
    mean_time = np.mean(all_times)
    print(min_time, "min time")
    print(max_time, "max time")
    print(mean_time, "mean time")


def determine_wallclock_er(dataset_names):
    min_time, max_time, mean_time = None,None,0
    all_times = []



    for dataset in dataset_names:
        print(dataset, "dataset")
        for i in range(1,6):

            #get featurs and labels
            features = np.load(f"datasets/features_{dataset}_{i}.npy")
            labels = np.load(f"datasets/labels_{dataset}_{i}.npy")

            #turn them in pytorch format
            features = torch.from_numpy(features).float()
            labels = torch.from_numpy(labels).long()

            #get model
            model = Net_10_2()
            #load model weights
            model.load_state_dict(torch.load(f"model_weights/{dataset}_{i}.pth"))

            for i in range(0,10000):

                #start timing
                query_time = time.time()
                
                #forward pass only
                with torch.no_grad():
                    x = model(features[i])

                    #get predictions
                    predicted = torch.argmax(torch.softmax(x, dim=0), dim=0)
                    result = int(not labels[i] == predicted)
                
                #end timing
                query_time = time.time() - query_time

                
                #add time
                all_times.append(query_time)
                if min_time == None or query_time < min_time:
                    print(min_time)
                    min_time = query_time
                    print(min_time)

                if max_time == None or query_time > max_time:
                    max_time = query_time
    
    mean_time = np.mean(all_times)
    print(min_time*1000, "min time")
    print(max_time*1000, "max time")
    print(mean_time*1000, "mean time")

def main():

    #params

    #dataset_names = ["SEA_0_1", "SEA_0_2", "SEA_0_3", "SEA_1_2", "SEA_1_3", "SEA_2_3"]
    dataset_names = ["HYP_001"]
    determine_wallclock(dataset_names)
    #determine_wallclock_er(dataset_names)



if __name__ == '__main__':
    main()