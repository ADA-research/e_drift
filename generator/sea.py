import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler




def generate_data(feature_name, label_name, seed, samples, drift_idx):
    #set seed
    np.random.seed(seed=seed)

    
    #generate features
    X = np.random.rand(samples, 3)
    y = []

    for idx, instance in enumerate(X):

        if idx>=drift_idx:

            # if instance[2]> 0.8:
            #     y.append(1)
            # else:
            #     y.append(0)
            y.append((instance[1] + instance[2] > 0.8).astype(int))
        
        else:
            y.append((instance[0] + instance[1] > 0.8).astype(int))
    
    return X,y
    
def save_data(features, labels, feature_name, label_name):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    features = np.array(features)
    labels = np.array(labels)

    np.save(feature_name, features)
    np.save(label_name, labels)



feature_name = "datasets_42/SEA_features_s_12_23.npy"
label_name = "datasets_42/SEA_labels_s_12_23.npy"
seed = 42
samples = 10000
drift_idx = 5000

features, labels = generate_data(feature_name, label_name, seed, samples, drift_idx)
save_data(features, labels, feature_name, label_name)
