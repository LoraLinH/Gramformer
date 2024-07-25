import re
import torch
import torch.nn.functional as F
import math


def generate_graph(y, k=0.2):
    with torch.no_grad():
        N, B, C = y.shape
        k = (int)(len(y) * k)
        y = y.permute(1,0,2)
        dist_matrix = torch.cdist(y, y) / (C ** 0.5)
        dist_matrix = dist_matrix[0] + torch.eye(y.size(1), dtype=dist_matrix.dtype, device=dist_matrix.device)*torch.max(dist_matrix)
        # dist_matrix = dist_matrix[:, :N]
        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        graph_matrix = torch.zeros_like(dist_matrix)
        ones = torch.ones_like(dist_matrix)
        graph_matrix.scatter_(-1, index_nearest, ones)
        
        return graph_matrix[:N,:N]