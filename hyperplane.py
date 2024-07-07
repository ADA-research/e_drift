import numpy as np
from river.datasets import synth

import random

class Hyperplane():

    def __init__(self, seed, features, drift_features, noise, mag_change, sigma):
        self.seed = seed
        self.features = features
        self.drift_features = drift_features
        self.noise = noise
        self.rng = random.Random(self.seed)
        self.mag_change = mag_change
        self.sigma = sigma
        self.weights = [self.rng.random() for _ in range(self.features)]
        self.change_direction = [1] * self.drift_features + [0] * (self.features - self.drift_features)

    def generate_drift(self):
        for i in range(self.drift_features):
            self.weights[i] += self.change_direction[i] * self.mag_change
            if (0.01 + self.rng.random() <= self.sigma):
                self.change_direction[i] *= -1
    
    def generate_points(self, number):
        print(self.weights)
        x_list, y_list =[], []
        for i in range(number):
            x = dict()
            sum_weights = np.sum(self.weights)
            sum_value = 0
            for i in range(self.features):
                x[i] = self.rng.random()
                sum_value+= self.weights[i] * x[i]
            

            y = 1 if sum_value>= sum_weights *0.5 else 0
            
            self.generate_drift()
        
            x_list.append(list(x.values()))
            y_list.append(y)
        return x_list, y_list

seed = 42
features = 2
drift_features = 2
noise = 0.0
mag_change = 0.000
sigma = 0.0

hyp = Hyperplane(seed = seed, features = features, drift_features = drift_features, noise = noise, mag_change = mag_change, sigma = sigma)

x,y = hyp.generate_points(5)
print(x, y)

seed = 0
dataset = synth.Hyperplane(seed=seed, n_features=2)