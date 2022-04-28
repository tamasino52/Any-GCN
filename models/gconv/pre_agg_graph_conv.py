from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreAggGraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj, bias=True):
        super(PreAggGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_pts = adj.size(1)
        self.W = nn.Parameter(torch.zeros(size=(self.n_pts, in_features, out_features), dtype=torch.float))

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
        h0 = torch.einsum('bjn,jnm->bjm', input, self.W)
        output = torch.matmul(self.adj, h0)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DecouplePreAggGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, decouple=True, bias=True):
        super(DecouplePreAggGraphConv, self).__init__()
        self.decouple = decouple
        self.in_features = in_features
        self.out_features = out_features
        self.n_pts = adj.size(1)
        self.W = nn.Parameter(torch.zeros(size=(2, self.n_pts, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        adj = self.adj.to(input.device)

        if self.decouple:
            h0 = torch.einsum('bjn,jnm->bjm', input, self.W[0])
            h1 = torch.einsum('bjn,jnm->bjm', input, self.W[1])
            E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
            output = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
        else:
            h0 = torch.einsum('bjn,jnm->bjm', input, self.W[0])
            output = torch.matmul(self.adj, h0)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'