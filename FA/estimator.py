import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
from math import pi


class QEstimator():
    def __init__(self, n_f, n_s, n_a, lr=0.05) -> None:
        # Different parameters of models used for different actions
        self.w, self.b = self.get_gaussian_wb(n_f, n_s)
        self.num_features = n_f
        self.models = []
        self.optims = []
        self.criterion = nn.MSELoss()
        for i in range(n_a):
            model = nn.Linear(self.num_features, 1)
            optim = SGD(model.parameters(), lr)
            self.optims.append(optim)
            self.models.append(model)

    def get_gaussian_wb(self, num_features, num_states, sigma=0.2):
        # Gaussian distribution of weights and biases
        torch.manual_seed(0)
        w = torch.randn((num_states, num_features)) * 1.0 / sigma
        b = torch.randn(num_features) * 2.0 * pi
        return w, b
    
    def get_feature(self, s):
        # Cosine Transform
        features = (2.0 / self.num_features) ** 0.5 * torch.cos(
                    torch.matmul(torch.tensor(s).float(), self.w) + self.b)
        return features
    
    def update(self, state, action, target):
        features = Variable(self.get_feature(state))
        y_pred = self.models[action](features)
        loss = self.criterion(y_pred, Variable(torch.Tensor([target])))
        self.optims[action].zero_grad()
        loss.backward()
        self.optims[action].step()
    
    def predict(self, state):
        # Compute Q-value from state using FA
        features = self.get_feature(state)
        with torch.no_grad():
            return torch.tensor([model(features) for model in self.models])


class NNEstimator():
    def __init__(self, n_f, n_s, n_a, n_h, lr=0.05) -> None:
        # Different parameters of models used for different actions
        self.w, self.b = self.get_gaussian_wb(n_f, n_s)
        self.num_features = n_f
        self.models = []
        self.optims = []
        self.criterion = nn.MSELoss()
        for i in range(n_a):
            # Change basic linear model to a 3 layer Linear Network
            model = nn.Sequential(nn.Linear(self.num_features, n_h), nn.ReLU(), nn.Linear(n_h, 1))
            optim = Adam(model.parameters(), lr)
            self.optims.append(optim)
            self.models.append(model)

    def get_gaussian_wb(self, num_features, num_states, sigma=0.2):
        # Gaussian distribution of weights and biases
        torch.manual_seed(0)
        w = torch.randn((num_states, num_features)) * 1.0 / sigma
        b = torch.randn(num_features) * 2.0 * pi
        return w, b
    
    def get_feature(self, s):
        # Cosine Transform
        features = ((2.0 / self.num_features) ** 0.5) * torch.cos(
                    torch.matmul(torch.tensor(s).float(), self.w) + self.b)
        return features
    
    def update(self, state, action, target):
        features = Variable(self.get_feature(state))
        y_pred = self.models[action](features)
        loss = self.criterion(y_pred, Variable(torch.Tensor([target])))
        self.optims[action].zero_grad()
        loss.backward()
        self.optims[action].step()
    
    def predict(self, state):
        # Compute Q-value from state using FA
        features = self.get_feature(state)
        with torch.no_grad():
            return torch.tensor([model(features) for model in self.models])
