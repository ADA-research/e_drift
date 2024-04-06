import numpy as np
from river import preprocessing, datasets


#generate one big class
class Dataset():
    def __init__(self, noise=0.0, seed=42):
        self.seed = seed
        self.noise = noise

    
    def printseed(self):
        print(self.seed)


#generate small classes per dataset?
class SEA(Dataset):
    pass

class HYP(Dataset):
    pass

