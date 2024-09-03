#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
class GraphAttention(nn.Module):
    def __init__(self, hidden_dim, out_dim, bias=True):
        super(GraphAttention, self).__init__()
        # self.n_head = n_head
        #hidden_dim=64, zdim=32
        self.f_in = hidden_dim
        self.f_out = out_dim
        self.w = nn.Parameter(torch.Tensor(hidden_dim, out_dim))
        self.a_src = nn.Parameter(torch.Tensor(out_dim, 1))
        self.a_dst = nn.Parameter(torch.Tensor(out_dim, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0)  #attn_dropout
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h, self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        # attn = attn_src + attn_dst.permute(0, 1, 2)
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1)
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn


# class GraphAttention(nn.Module):
#     def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
#         super(GraphAttention, self).__init__()
#         self.n_head = n_head
#         self.f_in = f_in
#         self.f_out = f_out
#         self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
#         self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
#         self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))
#
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(attn_dropout)
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(f_out))
#             nn.init.constant_(self.bias, 0)
#         else:
#             self.register_parameter("bias", None)
#
#         nn.init.xavier_uniform_(self.w, gain=1.414)
#         nn.init.xavier_uniform_(self.a_src, gain=1.414)
#         nn.init.xavier_uniform_(self.a_dst, gain=1.414)
#
#     def forward(self, h):
#         print(h.size())
#         bs, n = h.size()[:2]
#         # print(h.size())
#         h_prime = torch.matmul(h.unsqueeze(1), self.w)
#         attn_src = torch.matmul(h_prime, self.a_src)
#         attn_dst = torch.matmul(h_prime, self.a_dst)
#         attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
#             0, 1, 3, 2
#         )
#         attn = self.leaky_relu(attn)
#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         output = torch.matmul(attn, h_prime)
#         if self.bias is not None:
#             return output + self.bias, attn
#         else:
#             return output, attn