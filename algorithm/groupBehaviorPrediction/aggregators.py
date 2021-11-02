import torch
import math as m

eps = 1e-5

def aggregate_mean(X):
    X_mean = torch.mean(X, dim=0, keepdim=True)
    return X_mean

def aggregate_max(X):
    X_max = torch.max(X, dim=0, keepdim=True)
    return X_max[0]

def aggregate_min(X):
    X_min = torch.min(X, dim=0, keepdim=True)
    return X_min[0]

def aggregate_std(X):
    X_mean_square = torch.mean(X**2, dim=0, keepdim=True)
    X_square_mean = torch.mean(X, dim=0, keepdim=True)**2
    X_var = torch.relu(X_mean_square-X_square_mean)
    X_std = torch.sqrt(X_var + eps)
    return X_std

def aggregate_var(X):
    X_mean_square = torch.mean(X**2, dim=0, keepdim=True)
    X_square_mean = torch.mean(X, dim=0, keepdim=True)**2
    X_var = torch.relu(X_mean_square-X_square_mean)
    return X_var

def aggregate_sum(X):
    X_sum = torch.sum(X, dim=0, keepdim=True)
    return X_sum

def aggregate_softmax(X):
    X_softmax = torch.softmax(X, dim=0)
    X_sum_softmax = torch.sum(torch.mul(X_softmax, X), dim=0, keepdim=True)
    return X_sum_softmax

def aggregate_softmin(X):
    return -aggregate_softmax(-X)

AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var, 'softmax': aggregate_softmax, 'softmin': aggregate_softmin}
