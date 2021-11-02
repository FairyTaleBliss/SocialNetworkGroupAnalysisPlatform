import torch
import torch.nn as nn
from .aggregators import AGGREGATORS
from .scalers import SCALERS
from .layers import MLP

class PmaLayer(nn.Module):
    def __init__(self, embed_size, avg_m, aggregators, scalers, num_layers, dropout=0.):
        super(PmaLayer, self).__init__()

        self.embed_size = embed_size
        self.avg_m = avg_m          # 平均群体成员个数

        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.mlp = MLP(in_size=len(aggregators) * len(scalers) * embed_size, hidden_size=embed_size, out_size=embed_size,
                       layers=num_layers, dropout=dropout)

    def forward(self, X, m):
        X_aggre = torch.cat([aggregate(X) for aggregate in self.aggregators], dim=1)
        X_scale = torch.cat([scale(X_aggre, m=m, avg_m=self.avg_m) for scale in self.scalers], dim=1)
        X_out = self.mlp(X_scale)
        return X_out

