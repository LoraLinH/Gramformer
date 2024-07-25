import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from torch.nn import MultiheadAttention
from models.graph_attention import GraphMultiheadAttention
from models.utils import generate_graph


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, h_dim, nhead, norm, topk, usenum):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.topk = topk
        self.usenum = usenum
        self.near_embeddings = nn.Embedding(usenum, h_dim)
        self.pe_layer = nn.Sequential(
            nn.Conv2d(h_dim, h_dim // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim // 2, nhead, 3, 1, 1), # for quantize
            nn.Sigmoid()
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, 
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        x = src
        bs, c, h, w = x.shape
        pe = self.pe_layer(x)
        graph_pos = pe.flatten(2)
        graph_pos = torch.abs(graph_pos.permute(2,0,1) - graph_pos.permute(0,2,1))
        x = x.flatten(2).permute(2, 0, 1)
        attn = []
        for i, layer in enumerate(self.layers):
            graph = generate_graph(x, self.topk)
            near = torch.sum(graph, dim=0)
            near = near / torch.max(near) * (self.usenum-1)
            # near = (near-torch.min(near))/(torch.max(near)-torch.min(near)) * (self.usenum-1)
            # near = torch.clamp(near, max=self.usenum-1)
            pos = self.near_embeddings(near.int())
            x, att = layer(x, graph_pos=graph_pos, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos.unsqueeze(1))
            attn.append(att)
        
        if self.norm is not None:
            x = self.norm(x)
        output = x.permute(1, 2, 0).view(bs, c, h, w)

        return output, attn

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = GraphMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, graph_pos: Optional[Tensor] = None, src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2, attn = self.self_attn(q, k, src, graph_pos=graph_pos, mask=src_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
