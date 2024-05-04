import numpy as np
from river import preprocessing
from river.datasets import synth


#generate one big class
class Dataset():
    def __init__(self, noise=0.0, seed_1=42, seed_2=42, normalization = False):
        self.seed_1 = seed_1
        self.seed_2 = seed_2
        self.noise = noise
        self.normalization = normalization

    def generate_dataset(self, var_1: object, var_2: object, datasetname: str):
        """Generate features and labels with drift injected at timestep 5000."""
        features, labels = [], []
        #initialze normalizer
        if self.normalization:
            scaler = preprocessing.MinMaxScaler()
        
        #generate data without drift
        for x, y in var_1.take(5000):

            if self.normalization:
                scaler.learn_one(x)
                x = scaler.transform_one(x)
            features.append(list(x.values()))
            labels.append(y)

        #generate data with drift
        for x, y in var_2.take(5000):

            if self.normalization:
                scaler.learn_one(x)
                x = scaler.transform_one(x)
            features.append(list(x.values()))
            labels.append(y)
    
        
        #save features and labels
        features = np.array(features)
        labels = np.array(labels)
        np.save(f"datasets/features_{datasetname}.npy", features)
        np.save(f"datasets/labels_{datasetname}.npy", labels)

#generate small classes per dataset?
class SEA(Dataset):
    def __init__(self, variant_1, variant_2, noise=0.0, seed_1=42, seed_2=42, normalization=False):
        super().__init__(noise, seed_1, seed_2, normalization)
        self.var_1 = synth.SEA(variant = variant_1, noise = noise, seed = seed_1)
        self.var_2 = synth.SEA(variant = variant_2, noise = noise, seed = seed_2)

class HYP(Dataset):
    def __init__(self, n_features, n_drift_features, mag_change, sigma=0.0, noise=0.0, seed_1=42, seed_2=42, normalization=False):
        super().__init__(noise, seed_1, seed_2, normalization)
        self.var_1 = synth.Hyperplane(n_features=n_features, n_drift_features=n_drift_features, 
                                          mag_change=0.0, sigma=sigma, noise_percentage=noise, seed=seed_1)
        self.var_2 = synth.Hyperplane(n_features=n_features, n_drift_features=n_drift_features, 
                                          mag_change=mag_change, sigma=sigma, noise_percentage=noise, seed=seed_2)

