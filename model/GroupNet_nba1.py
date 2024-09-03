from random import sample
from tkinter import TRUE
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from model.utils import initialize_weights
from .MS_HGNN_batch import MS_HGNN_oridinary,MS_HGNN_hyper, MLP, MS_HGNN_hyper_angle
from .GATWeight import GraphAttention
import math
import networkx as nx
import time
import multiprocessing

class DecomposeBlock(nn.Module):
    '''
    Balance between reconstruction task and prediction task.
    '''
    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()
        # * HYPER PARAMETERS
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)
        
        self.decoder_y = MLP(dim_embedding_key + input_dim, future_len * 2, hidden_size=(512, 256))
        self.decoder_x = MLP(dim_embedding_key + input_dim, past_len * 2, hidden_size=(512, 256))

        self.relu = nn.ReLU()

        # kaiming initialization
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)


    def forward(self, x_true, x_hat, f):
        '''
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, D

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        '''
        x_ = x_true - x_hat
        x_ = torch.transpose(x_, 1, 2)
        
        past_embed = self.relu(self.conv_past(x_))
        past_embed = torch.transpose(past_embed, 1, 2)

        _, state_past = self.encoder_past(past_embed)
        state_past = state_past.squeeze(0)

        input_feat = torch.cat((f, state_past), dim=1)

        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)
        
        return x_hat_after, y_hat

class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())        

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a,1,1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N,T,D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x) #(N,T,D)

class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)
            
        self.input_fc = nn.Linear(in_dim, self.model_dim)
        self.input_fc2 = nn.Linear(self.model_dim*args.past_length,self.model_dim)
        categ =11
        self.input_fc3 = nn.Linear(self.model_dim+categ,self.model_dim)

        self.interaction = MS_HGNN_oridinary(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1
        )

        # self.Grapinteraction = GraphAttention(
        #     hidden_dim=64,
        #     out_dim=32,
        #     bias=True)

        if len(args.hyper_scales) > 0:
            self.interaction_hyper_angle = MS_HGNN_hyper_angle(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0]
            )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0]
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[1]
            )

        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[2]
            )
        if len(args.hyper_scales) > 3:
            self.interaction_hyper4 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[3]
            )
        if len(args.hyper_scales) > 4:
            self.interaction_hyper5 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[4]
            )

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)
    
    # def add_category(self,x):
    #     B = x.shape[0]
    #     N = x.shape[1]
    #     category = torch.zeros(N,3).type_as(x)
    #     category[0:5,0] = 1
    #     category[5:10,1] = 1
    #     category[10,2] = 1
    #     category = category.repeat(B,1,1)
    #     x = torch.cat((x,category),dim=-1)
    #     return x

    def add_category(self, x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N, N).type_as(x)
        for i in range(0, N):
            category[i, i] = 1
        category = category.repeat(B, 1, 1)
        x = torch.cat((x, category), dim=-1)
        return x

    # def group_complete(self, weight):
    #     groups = []
    #     w = weight.cpu().numpy()
    #     G = nx.from_numpy_array(w)
    #     complete_subgraphs = list(nx.find_cliques(G))
    #     for i, subgraph in enumerate(complete_subgraphs):
    #         # print(f'Group {i + 1}: {subgraph}')
    #         groups.append(subgraph)
    #
    #     return groups

    def line_embedding(self, feat_one):
        d = feat_one.size()[-1]
        w1 = nn.Parameter(torch.Tensor(1, d, 64))
        w = nn.init.xavier_uniform_(w1, gain=1.414)
        feat = torch.matmul(feat_one, w.cuda())

        return feat

    def group_complete(self, weight, ftraj, num_ped):
        # groups = []
        w = weight.cpu().numpy()
        G = nx.from_numpy_array(w)
        complete_subgraphs = list(nx.find_cliques(G))
        features = []
        for i, subgraph in enumerate(complete_subgraphs):
            # print(f'Group {i + 1}: {subgraph}')
            # groups.append(subgraph)
            feat_c = weight
            num_diff = set(num_ped).difference(set(subgraph))
            feat_c[list(num_diff)] = 0
            feat_c[:, list(num_diff)] = 0

            self.args.hyper_scales = [len(subgraph)]
            ftraj_inter_hyper, _ = self.interaction_hyper_angle(ftraj, torch.unsqueeze(feat_c, dim=0))
            ftraj_inter_hyper[:, list(num_diff), :] = ftraj[:, list(num_diff), :]
            features.append(ftraj_inter_hyper)
        feat_one = self.line_embedding(torch.cat(features, dim=-1))

        return feat_one

    def divide_group_adap(self, feat_corr, batch_size, agent_num, ftraj_input, ftraj_inter):
        # bs, n, _= feat_corr.size()
        zeros = torch.zeros([batch_size, agent_num, agent_num]).type_as(feat_corr)
        ones = torch.ones([batch_size, agent_num, agent_num]).type_as(feat_corr)
        feat_corr = torch.where(feat_corr > 0, ones, zeros)
        num_ped = list(range(agent_num))

        feat_bs = []
        for i in range(batch_size):
            feat_co = feat_corr[i]
            ftraj = torch.unsqueeze(ftraj_input[i], dim=0)
            feat_one = self.group_complete(feat_co, ftraj, num_ped)
            feat_bs.append(feat_one)
        output_feature = torch.cat((ftraj_input, ftraj_inter, torch.cat(feat_bs, dim=0)), dim=-1)
        output_feature = output_feature.view(batch_size * agent_num, -1)

        return output_feature


    def forward(self, inputs,batch_size, agent_num):
        length = inputs.shape[1]

        tf_in = self.input_fc(inputs).view(batch_size*agent_num, length, self.model_dim)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*agent_num)
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)
  
        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, length*self.model_dim))
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))

        query_input = F.normalize(ftraj_input,p=2,dim=2)
        feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))

        # ## calculate attention among agents by zyz
        # _, nodes_attn = self.Grapinteraction(ftraj_input)
        # feat_corr =nodes_attn
        # ## calculate attention among agents by zyz

        # # # ## calculate attention among agents by zyz
        ZEROS = torch.zeros(agent_num, feat_corr.shape[-1]).type_as(feat_corr)
        for i in range(0, batch_size):
            a = feat_corr[i].min()
            if a < 0.4:
                feat_corr[i] = torch.where(feat_corr[i] < 0.4, ZEROS, feat_corr[i])
            elif 0.4 < a < 0.6:
                feat_corr[i] = torch.where(feat_corr[i] < (a + 0.1), ZEROS, feat_corr[i])
            else:
                feat_corr[i] = torch.where(feat_corr[i] < (a + 0.03), ZEROS, feat_corr[i])
        ######

        ftraj_inter,_ = self.interaction(ftraj_input)

        output_feature = self.divide_group_adap(feat_corr, batch_size, agent_num, ftraj_input, ftraj_inter)

        # if len(self.args.hyper_scales) > 0:
        #     ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 1:
        #     ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 2:
        #     ftraj_inter_hyper3,_ = self.interaction_hyper3(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 3:
        #     ftraj_inter_hyper4,_ = self.interaction_hyper4(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 4:
        #     ftraj_inter_hyper5,_ = self.interaction_hyper5(ftraj_input,feat_corr)
        #
        # if len(self.args.hyper_scales) == 0:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter),dim=-1)
        # if len(self.args.hyper_scales) == 1:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper),dim=-1)
        # elif len(self.args.hyper_scales) == 2:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        # elif len(self.args.hyper_scales) == 3:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3),dim=-1)
        # elif len(self.args.hyper_scales) == 4:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3,ftraj_inter_hyper4),dim=-1)
        # elif len(self.args.hyper_scales) == 5:
        #     final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2,
        #                                ftraj_inter_hyper3, ftraj_inter_hyper4,ftraj_inter_hyper5), dim=-1)
        #
        # output_feature = final_feature.view(batch_size*agent_num,-1)
        return output_feature

class FutureEncoder(nn.Module):
    def __init__(self, args,in_dim=4):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim

        self.input_fc = nn.Linear(in_dim, self.model_dim)
        scale_num = 2 + len(self.args.hyper_scales)
        self.input_fc2 = nn.Linear(self.model_dim*self.args.future_length, self.model_dim)
        categ = 11
        self.input_fc3 = nn.Linear(self.model_dim+categ, self.model_dim)

        self.interaction = MS_HGNN_oridinary(
            embedding_dim=16,
            h_dim=self.model_dim,
            mlp_dim=64,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=1,
            vis=False
        )

        # self.Grapinteraction = GraphAttention(
        #     hidden_dim=64,
        #     out_dim=32,
        #     bias=True)

        if len(args.hyper_scales) > 0:
            self.interaction_hyper_angle = MS_HGNN_hyper_angle(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0]
            )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[0],
                vis=False
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[1],
                vis=False
            )
        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[2],
                vis=False
            )
        if len(args.hyper_scales) > 3:
            self.interaction_hyper4 = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[3],
                vis=False
            )
        if len(args.hyper_scales) > 4:
            self.interaction_hyper5 = MS_HGNN_hyper(
                embedding_dim=16,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=1,
                scale=args.hyper_scales[4],
                vis=False
            )

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

        self.out_mlp = MLP2(scale_num*2*self.model_dim, [128], 'relu')
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.args.zdim)
        initialize_weights(self.qz_layer.modules())

    # def add_category(self,x):
    #     B = x.shape[0]
    #     N = x.shape[1]
    #     category = torch.zeros(N,3).type_as(x)
    #     category[0:5,0] = 1
    #     category[5:10,1] = 1
    #     category[10,2] = 1
    #     category = category.repeat(B,1,1)
    #     x = torch.cat((x,category),dim=-1)
    #     return x

    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N, N).type_as(x)
        for i in range(0, N):
            category[i, i] = 1
        category = category.repeat(B, 1, 1)
        x = torch.cat((x, category), dim=-1)
        return x

    # def group_complete(self, weight):
    #     groups = []
    #     w = weight.cpu().numpy()
    #     G = nx.from_numpy_array(w)
    #     complete_subgraphs = list(nx.find_cliques(G))
    #     for i, subgraph in enumerate(complete_subgraphs):
    #         # print(f'Group {i + 1}: {subgraph}')
    #         groups.append(subgraph)
    #
    #     return groups

    def line_embedding(self, feat_one):
        d = feat_one.size()[-1]
        w1 = nn.Parameter(torch.Tensor(1, d, 64))
        w = nn.init.xavier_uniform_(w1, gain=1.414)
        feat = torch.matmul(feat_one, w.cuda())

        return feat

    def group_complete0(self, weight, ftraj, num_ped):
        # groups = []
        w = weight.cpu().numpy()
        G = nx.from_numpy_array(w)
        complete_subgraphs = list(nx.find_cliques(G))
        features = []
        for i, subgraph in enumerate(complete_subgraphs):
            # print(f'Group {i + 1}: {subgraph}')
            # groups.append(subgraph)
            feat_c = weight
            num_diff = set(num_ped).difference(set(subgraph))
            feat_c[list(num_diff)] = 0
            feat_c[:, list(num_diff)] = 0

            self.args.hyper_scales = [len(subgraph)]
            ftraj_inter_hyper, _ = self.interaction_hyper_angle(ftraj, torch.unsqueeze(feat_c, dim=0))
            ftraj_inter_hyper[:, list(num_diff), :] = ftraj[:, list(num_diff), :]
            features.append(ftraj_inter_hyper)
        feat_one = self.line_embedding(torch.cat(features, dim=-1))

        return feat_one

    def divide_group_adap0(self, feat_corr, batch_size, agent_num, ftraj_input, ftraj_inter):
        # bs, n, _= feat_corr.size()
        zeros = torch.zeros([batch_size, agent_num, agent_num]).type_as(feat_corr)
        ones = torch.ones([batch_size, agent_num, agent_num]).type_as(feat_corr)
        feat_corr = torch.where(feat_corr>0, ones, zeros)
        num_ped = list(range(agent_num))

        feat_bs = []
        for i in range(batch_size):
            feat_co = feat_corr[i]
            ftraj = torch.unsqueeze(ftraj_input[i], dim=0)
            feat_one = self.group_complete0(feat_co, ftraj, num_ped)
            feat_bs.append(feat_one)
        output_feature = torch.cat((ftraj_input, ftraj_inter, torch.cat(feat_bs, dim=0)), dim=-1)
        output_feature = output_feature.view(batch_size * agent_num, -1)

        return output_feature

    # def divide_group_adap(self, feat_corr, batch_size, agent_num, ftraj_input, ftraj_inter):
    #     # bs, n, _= feat_corr.size()
    #     zeros = torch.zeros([batch_size, agent_num, agent_num]).type_as(feat_corr)
    #     ones = torch.ones([batch_size, agent_num, agent_num]).type_as(feat_corr)
    #     feat_corr = torch.where(feat_corr>0, ones, zeros)
    #     num_ped = list(range(agent_num))
    #
    #     feat_bs = []
    #     for i in range(batch_size):
    #         feat_co = feat_corr[i]
    #         ftraj = torch.unsqueeze(ftraj_input[i], dim=0)
    #         groups = self.group_complete(feat_co)
    #         # groups = self.group_complete0(feat_co, ftraj)
    #         features = []
    #         for num_g in groups:
    #             feat_c = feat_co
    #             num_diff = set(num_ped).difference(set(num_g))
    #             feat_c[list(num_diff)] = 0
    #             feat_c[:, list(num_diff)] = 0
    #
    #             self.args.hyper_scales = [len(num_g)]
    #             ftraj_inter_hyper, _ = self.interaction_hyper_angle(ftraj, torch.unsqueeze(feat_c, dim=0))
    #             ftraj_inter_hyper[:, list(num_diff), :] = ftraj[:, list(num_diff), :]
    #             features.append(ftraj_inter_hyper)
    #         feat_one = self.line_embedding(torch.cat(features, dim=-1))
    #         feat_bs.append(feat_one)
    #     output_feature = torch.cat((ftraj_input, ftraj_inter, torch.cat(feat_bs, dim=0)), dim=-1)
    #     output_feature = output_feature.view(batch_size * agent_num, -1)
    #
    #     return output_feature
    #
    # def divide_group_adap1(self, feat_corr, batch_size, agent_num, ftraj_input, ftraj_inter):
    #     bs, n, _= feat_corr.size()
    #     zeros = torch.zeros([bs, n, n]).type_as(feat_corr)
    #     ones = torch.ones([bs, n, n]).type_as(feat_corr)
    #     feat_corr = torch.where(feat_corr>0, ones, zeros)
    #     num_ped = [i for i in range(n)]
    #
    #     feat_bs = []
    #     for i in range(bs):
    #         feat_co = feat_corr[i]
    #         groups = self.group_complete(feat_co)
    #         # num_groups = len(groups)  8
    #         features = []
    #         for num_g in groups:
    #             feat_c = feat_co
    #             num_diff = set(num_ped).difference(set(num_g))
    #             #set as 0 and not remove for the index in num_diff
    #             # feat_c[[i for i in num_diff], :] = 0
    #             # feat_c[:, [j for j in num_diff]] = 0
    #             for i in num_diff:
    #                 feat_c[i, :] = 0
    #                 feat_c[:, i] = 0
    #
    #             self.args.hyper_scales = [len(num_g)]
    #             ftraj = torch.unsqueeze(ftraj_input[i], dim=0)
    #             # feat_c = torch.unsqueeze(feat_c, dim=0)
    #             # ftraj_inter_hyper, _ = self.interaction_hyper_angle(ftraj, feat_c)
    #             ftraj_inter_hyper, _ = self.interaction_hyper_angle(ftraj, torch.unsqueeze(feat_c, dim=0))
    #             # ftraj_inter_hyper[:, [i for i in num_diff], :] = ftraj[:, [i for i in num_diff], :]
    #             for i in num_diff:
    #                 ftraj_inter_hyper[:, i, :] = ftraj[:, i, :]
    #             features.append(ftraj_inter_hyper)
    #         # features = torch.cat([fea for fea in features], dim=-1)
    #         # features = torch.cat(features, dim=-1)
    #         # feat_one = self.line_embedding(features)
    #         feat_one = self.line_embedding(torch.cat(features, dim=-1))
    #         feat_bs.append(feat_one)
    #     # feat_bs = torch.cat([fea for fea in feat_bs], dim=0)
    #     # final_feature = torch.cat(feat_bs, dim=0)
    #     # output_feature = torch.cat((ftraj_input, ftraj_inter, final_feature), dim=-1)
    #     output_feature = torch.cat((ftraj_input, ftraj_inter, torch.cat(feat_bs, dim=0)), dim=-1)
    #     output_feature = output_feature.view(batch_size * agent_num, -1)
    #
    #     return output_feature


    def forward(self, inputs, batch_size,agent_num,past_feature):
        length = inputs.shape[1]
        agent_num = 11
        tf_in = self.input_fc(inputs).view(batch_size*agent_num, length, self.model_dim)

        tf_in_pos = self.pos_encoder(tf_in, num_a=batch_size*agent_num)
        tf_in_pos = tf_in_pos.view(batch_size, agent_num, length, self.model_dim)

        ftraj_input = self.input_fc2(tf_in_pos.contiguous().view(batch_size, agent_num, -1))
        ftraj_input = self.input_fc3(self.add_category(ftraj_input))

        query_input = F.normalize(ftraj_input,p=2,dim=2)
        feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))

        # ## calculate attention among agents by zyz
        # _, nodes_attn = self.Grapinteraction(ftraj_input)
        # feat_corr = nodes_attn
        # ## calculate attention among agents by zyz

        # # # ## calculate attention among agents by zyz
        ZEROS = torch.zeros(agent_num, feat_corr.shape[-1]).type_as(feat_corr)
        for i in range(0, batch_size):
            a = feat_corr[i].min()
            if a < 0.4:
                feat_corr[i] = torch.where(feat_corr[i] < 0.4, ZEROS, feat_corr[i])
            elif 0.4 < a < 0.6:
                feat_corr[i] = torch.where(feat_corr[i] < (a + 0.1), ZEROS, feat_corr[i])
            else:
                feat_corr[i] = torch.where(feat_corr[i] < (a + 0.03), ZEROS, feat_corr[i])
        ######

        ftraj_inter, _ = self.interaction(ftraj_input)

        final_feature = self.divide_group_adap0(feat_corr, batch_size, agent_num, ftraj_input, ftraj_inter)


        # if len(self.args.hyper_scales) > 0:
        #     ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 1:
        #     ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 2:
        #     ftraj_inter_hyper3,_ = self.interaction_hyper3(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 3:
        #     ftraj_inter_hyper4,_ = self.interaction_hyper4(ftraj_input,feat_corr)
        # if len(self.args.hyper_scales) > 4:
        #     ftraj_inter_hyper5,_ = self.interaction_hyper5(ftraj_input,feat_corr)
        #
        # if len(self.args.hyper_scales) == 0:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter),dim=-1)
        # if len(self.args.hyper_scales) == 1:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper),dim=-1)
        # elif len(self.args.hyper_scales) == 2:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        # elif len(self.args.hyper_scales) == 3:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3),dim=-1)
        # elif len(self.args.hyper_scales) == 4:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3,ftraj_inter_hyper4),dim=-1)
        # elif len(self.args.hyper_scales) == 5:
        #     final_feature = torch.cat((ftraj_input,ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3,ftraj_inter_hyper4,ftraj_inter_hyper5),dim=-1)



        # final_feature = final_feature.view(batch_size*agent_num,-1)

        h = torch.cat((past_feature,final_feature),dim=-1)
        h = self.out_mlp(h)
        q_z_params = self.qz_layer(h)
        return q_z_params

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.decode_way = 'RES'
        scale_num = 2 + len(self.args.hyper_scales)
        
        self.num_decompose = args.num_decompose
        input_dim = scale_num*self.model_dim+self.args.zdim
        self.past_length = self.args.past_length
        self.future_length = self.args.future_length

        self.decompose = nn.ModuleList([DecomposeBlock(self.args.past_length, self.args.future_length, input_dim) for _ in range(self.num_decompose)])

    def forward(self, past_feature, z, batch_size_curr, agent_num_perscene, past_traj, cur_location, sample_num, mode='train'):
        agent_num = batch_size_curr * agent_num_perscene
        past_traj_repeat = past_traj.repeat_interleave(sample_num, dim=0)
        past_feature = past_feature.view(-1,sample_num,past_feature.shape[-1])

        z_in = z.view(-1, sample_num, z.shape[-1])

        hidden = torch.cat((past_feature,z_in),dim=-1)
        hidden = hidden.view(agent_num*sample_num,-1)
        x_true = past_traj_repeat.clone() #torch.transpose(pre_motion_scene_norm, 0, 1)

        x_hat = torch.zeros_like(x_true)
        batch_size = x_true.size(0)
        prediction = torch.zeros((batch_size, self.future_length, 2)).cuda()
        reconstruction = torch.zeros((batch_size, self.past_length, 2)).cuda()

        for i in range(self.num_decompose):
            x_hat, y_hat = self.decompose[i](x_true, x_hat, hidden)
            prediction += y_hat
            reconstruction += x_hat
        norm_seq = prediction.view(agent_num*sample_num,self.future_length,2)
        recover_pre_seq = reconstruction.view(agent_num*sample_num,self.past_length,2)

        # norm_seq = norm_seq.permute(2,0,1,3).view(self.future_length, agent_num * sample_num,2)

        cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        out_seq = norm_seq + cur_location_repeat # (agent_num*sample_num,self.past_length,2)
        if mode == 'inference':
            out_seq = out_seq.view(-1,sample_num,*out_seq.shape[1:]) # (agent_num,sample_num,self.past_length,2)
        return out_seq,recover_pre_seq
        
class GroupNet(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        self.args = args

        # models
        scale_num = 2 + len(self.args.hyper_scales)
        self.past_encoder = PastEncoder(args)
        self.pz_layer = nn.Linear(scale_num*self.args.hidden_dim, 2 * self.args.zdim)
        if args.learn_prior:
            initialize_weights(self.pz_layer.modules())
        self.future_encoder = FutureEncoder(args)
        self.decoder = Decoder(args)
        self.param_annealers = nn.ModuleList()

    def set_device(self, device):
        self.device = device
        self.to(device)
    
    def calculate_loss_pred(self,pred,target,batch_size):
        loss = (target-pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss
    
    def calculate_loss_kl(self,qz_distribution,pz_distribution,batch_size,agent_num,min_clip):
        loss = qz_distribution.kl(pz_distribution).sum()
        loss /= (batch_size * agent_num)
        loss_clamp = loss.clamp_min_(min_clip)
        return loss_clamp

    def calculate_loss_recover(self,pred,target,batch_size):
        loss = (target-pred).pow(2).sum()
        loss /= batch_size
        loss /= pred.shape[1]
        return loss
    
    def calculate_loss_diverse(self,pred,target,batch_size):
        diff = target.unsqueeze(1) - pred
        avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        loss = avg_dist.min(dim=1)[0]
        loss = loss.mean() 
        return loss

    def forward(self,data):
        device = self.device
        batch_size = data['past_traj'].shape[0]
        agent_num = data['past_traj'].shape[1]
        
        past_traj = data['past_traj'].view(batch_size*agent_num,self.args.past_length,2).to(device).contiguous()
        future_traj = data['future_traj'].view(batch_size*agent_num,self.args.future_length,2).to(device).contiguous()

        past_vel = past_traj[:,1:] - past_traj[:,:-1, :]
        past_vel = torch.cat([past_vel[:,[0]], past_vel], dim=1)

        future_vel = future_traj - torch.cat([past_traj[:,[-1]], future_traj[:,:-1, :]],dim=1)
        cur_location = past_traj[:,[-1]]

        inputs = torch.cat((past_traj,past_vel),dim=-1)
        inputs_for_posterior = torch.cat((future_traj,future_vel),dim=-1)

        past_feature = self.past_encoder(inputs,batch_size,agent_num)
        qz_param = self.future_encoder(inputs_for_posterior,batch_size,agent_num,past_feature)

        ### q dist ###
        if self.args.ztype == 'gaussian':
            qz_distribution = Normal(params=qz_param)
        else:
            ValueError('Unknown hidden distribution!')
        qz_sampled = qz_distribution.rsample()

        ### p dist ###
        if self.args.learn_prior:
            pz_param = self.pz_layer(past_feature)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=pz_param)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(mu=torch.zeros(past_feature.shape[0], self.args.zdim).to(past_traj.device), 
                                        logvar=torch.zeros(past_feature.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')


        ### use q ###
        # z = qz_sampled
        pred_traj,recover_traj = self.decoder(past_feature,qz_sampled,batch_size,agent_num,past_traj,cur_location,sample_num=1)
        loss_pred = self.calculate_loss_pred(pred_traj,future_traj,batch_size)

        loss_recover = self.calculate_loss_recover(recover_traj,past_traj,batch_size)
        loss_kl = self.calculate_loss_kl(qz_distribution,pz_distribution,batch_size,agent_num,self.args.min_clip)
        

        ### p dist for best 20 loss ###
        sample_num = 20
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device), 
                                        logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        # z = pz_sampled

        diverse_pred_traj,_ = self.decoder(past_feature_repeat,pz_sampled,batch_size,agent_num,past_traj,cur_location,sample_num=20,mode='inference')
        loss_diverse = self.calculate_loss_diverse(diverse_pred_traj,future_traj,batch_size)
        total_loss = loss_pred + loss_recover + loss_kl+ loss_diverse

        return total_loss, loss_pred.item(), loss_recover.item(), loss_kl.item(), loss_diverse.item()

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def inference(self, data):
        device = self.device
        batch_size = data['past_traj'].shape[0]
        agent_num = data['past_traj'].shape[1]
        
        past_traj = data['past_traj'].view(batch_size*agent_num,self.args.past_length,2).to(device).contiguous()

        past_vel = past_traj[:,1:] - past_traj[:,:-1, :]
        past_vel = torch.cat([past_vel[:,[0]], past_vel], dim=1)

        cur_location = past_traj[:,[-1]]

        inputs = torch.cat((past_traj,past_vel),dim=-1)

        past_feature = self.past_encoder(inputs,batch_size,agent_num)

        sample_num = 20
        if self.args.learn_prior:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            p_z_params = self.pz_layer(past_feature_repeat)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(params=p_z_params)
            else:
                ValueError('Unknown hidden distribution!')
        else:
            past_feature_repeat = past_feature.repeat_interleave(sample_num, dim=0)
            if self.args.ztype == 'gaussian':
                pz_distribution = Normal(mu=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device), 
                                        logvar=torch.zeros(past_feature_repeat.shape[0], self.args.zdim).to(past_traj.device))
            else:
                ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution.rsample()
        z = pz_sampled

        diverse_pred_traj,_ = self.decoder(past_feature_repeat,z,batch_size,agent_num,past_traj,cur_location,sample_num=self.args.sample_k,mode='inference')
        diverse_pred_traj = diverse_pred_traj.permute(1,0,2,3)
        return diverse_pred_traj
