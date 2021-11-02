import torch

def scale_identity(X, m, avg_m):
    return X

def scale_amplification(X, m, avg_m):
    scale = torch.log(m + 1.0) / torch.log(avg_m)
    X_scale = X * scale
    return X_scale

def scale_attenuation(X, m, avg_m):
    scale = torch.log(avg_m) / torch.log(m + 1.0)
    X_scale = X * scale
    return X_scale

SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation}