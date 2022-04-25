from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaGraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj, bias=True):
        super(VanillaGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(1, in_features, out_features), dtype=torch.float))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.diag = torch.diagflat(torch.diagonal(adj, 0))
        self.ex_diag = adj - self.diag

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        output = torch.matmul(self.adj, h0)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DecoupleVanillaGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, decouple=True, bias=True):
        super(DecoupleVanillaGraphConv, self).__init__()
        self.decouple = decouple
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(1, in_features, out_features), dtype=torch.float))
        self.T = nn.Parameter(torch.zeros(size=(1, in_features, out_features), dtype=torch.float)) if decouple else None

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.T.data, gain=1.414)

        self.adj = adj
        self.diag = torch.diagflat(torch.diagonal(adj, 0))
        self.ex_diag = adj - self.diag

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        if self.decouple:
            h0 = torch.matmul(input, self.W[0])
            t0 = torch.matmul(input, self.T[0])
            output = torch.matmul(self.ex_diag, h0) + torch.matmul(self.diag, t0)
        else:
            h0 = torch.matmul(input, self.W[0])
            output = torch.matmul(self.adj, h0)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'