import numpy as np

#network for 100 features
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
    
#network for 10 features
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
    
#default network for SEA dataset
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3,2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2,2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output