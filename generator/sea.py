import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler




def generate_data(feature_name, label_name, seed, samples, drift_idx):
    #set seed
    np.random.seed(seed=seed)

    
    #generate features
    X = np.random.rand(samples, 10)
    y = []

    for idx, instance in enumerate(X):

        if idx>=drift_idx:

            # if instance[2]> 0.8:
            #     y.append(1)
            # else:
            #     y.append(0)
            y.append((np.sum(instance[0:5]) > 2.8).astype(int))
        
        else:
            y.append((np.sum(instance[0:5]) > 2.5).astype(int))
    
    return X,y
    
def save_data(features, labels, feature_name, label_name):
    #scaler = MinMaxScaler()
    #features = scaler.fit_transform(features)

    features = np.array(features)
    labels = np.array(labels)

    values, counts = np.unique(labels, return_counts=True)
    print(values, counts)

    np.save(feature_name, features)
    np.save(label_name, labels)



feature_name = "datasets/SEA_features_s_10_l.npy"
label_name = "datasets/SEA_labels_s_10_l.npy"
seed = 42
samples = 10000
drift_idx = 5000

features, labels = generate_data(feature_name, label_name, seed, samples, drift_idx)
save_data(features, labels, feature_name, label_name)
