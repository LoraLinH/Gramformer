import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

def position(x):
    length, d_model = x.size()
    length = length+1
    pe = torch.zeros((length, d_model), device=x.device)
    # d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2, device=x.device)*-(math.log(10000.0)/d_model))
    pos = torch.arange(0, length, device=x.device).unsqueeze(1)
    pe[:, 0::2] = torch.sin(pos.float()*div_term)
    pe[:, 1::2] = torch.cos(pos.float()*div_term)
    return pe
    # pos_w = torch.arange(0, width, device=x.device).unsqueeze(1)
    # pos_h = torch.arange(0, height, device=x.device).unsqueeze(1)
    # pe[0:d_model:2,:,:]=torch.sin(pos_w*div_term).transpose(0,1).unsqueeze(1).repeat(1, height, 1)
    # pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    # pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, width, 1)
    # pe[d_model+1:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, width, 1)




class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_y = y_embed[:, :, :, None] / dim_t
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = pos_y.permute(0, 3, 1, 2).contiguous()

        return pos

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, graph_pos=None, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)


        # if graph_pos is not None:
        #     attn = F.softmax(attn, dim=-1)
        #     attn = attn * graph_pos.permute(2,1,0).unsqueeze(0)
        #     attn = attn / (torch.sum(attn, dim=-1, keepdim=True)+1e-9)
        #     attn = self.dropout(attn)
        # else:
        #     attn = self.dropout(F.softmax(attn, dim=-1))

        attno = self.dropout(F.softmax(attn, dim=-1))
        if graph_pos is not None:
            attn = attno * graph_pos.permute(2, 1, 0).unsqueeze(0)

        output = torch.matmul(attn, v)

        return output, attno
        # return output, attn / (torch.sum(attn, dim=-1, keepdim=True)+1e-6)

class GraphMultiheadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, graph_pos=None, mask=None):
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, graph_pos=graph_pos, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        q = q.permute(1,0,2)


        return q, attn