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

def get_misclassifications(dataset):

    for i in range(1, 6):

        training_mis, until_drift_mis = 0,0
        first_mis_after_drift = None

        data = pd.read_csv(f"results/{dataset}_{i}.csv")
        runtimes = data["runtime"].to_list()
        epsilons = data["epsilon"].to_list()

        for idx, (eps, runtime) in enumerate(zip(epsilons[0:], runtimes[100:])):
            idx+=0

            if runtime>0.0:

                if idx <2000:
                    training_mis+=1
                elif idx <5000:
                    until_drift_mis +=1
                else:
                    first_mis_after_drift = idx
                    break
        
        print(f"training mis, mis until driftpoint, first after drift point for dataset {dataset}_{i}:", 
              training_mis, until_drift_mis, first_mis_after_drift)



def main():

    dataset_names = ["SEA_0_1", "SEA_0_2", "SEA_0_3", "SEA_1_2", "SEA_1_3", "SEA_2_3", "HYP_001"]
    for dataset in dataset_names:
        get_misclassifications(dataset)


if __name__ == '__main__':
    main()