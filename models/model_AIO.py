import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layer_utils import *
from models.codebook import Codebook
from models.mingpt import GPT


class model_encdec(nn.Module):
    
    def __init__(self, settings, pretrained_model=None):
        super(model_encdec, self).__init__()

        self.name_model = 'AIO_autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = 64 
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.mode = settings["mode"]
        self.codebook_lantent_dim = 16
        self.num_codebook_vectors = settings["num_codebook_vectors"]
        self.gt_len = settings["gt_len"]   ###########

        assert self.mode in ['intention', 'transformer', 'addressor', 'trajectory'], 'WRONG MODE!'

        # LAYERS for different modes
        if self.mode == 'intention':
            self.abs_past_encoder = st_encoder(type=False)
            self.norm_past_encoder = st_encoder(type=False)
            self.norm_fut_encoder = st_encoder(type=False)
            self.norm_gt_encoder = st_encoder(type=True)    #############
            self.codebook = Codebook(self.num_codebook_vectors, self.codebook_lantent_dim, 0.5) #####
            # self.codebook_social = Codebook(256, 16, 0.5)
            # self.codebook_future = Codebook(256, 16, 0.5)

            self.res_past_encoder = st_encoder(type=False)
            self.res_gt_encoder = st_encoder(type=True)  ###############
            self.social_pooling_X = NmpNet(
                embedding_dim=self.dim_embedding_key,
                h_dim=self.dim_embedding_key,
                mlp_dim=1024,
                bottleneck_dim=self.dim_embedding_key,
                activation='relu',
                batch_norm=False,
                nmp_layers=2
            )
            # self.decoder = MLP(self.dim_embedding_key * 2, self.future_len * 2, hidden_size=(1024, 512, 1024))
            self.decoder_x = MLP(self.dim_embedding_key * 2 + 96, self.past_len * 2, hidden_size=(1024, 512, 1024))
            # self.decoder_2 = MLP(self.dim_embedding_key * 2, self.future_len * 2, hidden_size=(1024, 512, 1024))
            self.decoder_2_x = MLP(self.dim_embedding_key * 2 + 96, self.past_len * 2, hidden_size=(1024, 512, 1024))
            self.decoder_gt = MLP(self.dim_embedding_key * 2 + 96, self.gt_len * 2, hidden_size=(1024, 512, 1024))   ##########
            self.decoder_2_gt = MLP(self.dim_embedding_key * 2 + 96, self.gt_len * 2, hidden_size=(1024, 512, 1024))  ############
        else:
            ###############################################
            # 加载intention训练完成后各个网络的参数
            self.abs_past_encoder = pretrained_model.abs_past_encoder
            self.norm_past_encoder = pretrained_model.norm_past_encoder
            self.norm_fut_encoder = pretrained_model.norm_fut_encoder
            self.res_past_encoder = pretrained_model.res_past_encoder
            self.social_pooling_X = pretrained_model.social_pooling_X
            # self.decoder = pretrained_model.decoder
            self.decoder_x = pretrained_model.decoder_x
            # self.decoder_2 = pretrained_model.decoder_2
            self.decoder_2_x = pretrained_model.decoder_2_x
            self.norm_gt_encoder = pretrained_model.norm_gt_encoder        ###############
            self.res_gt_encoder = pretrained_model.res_gt_encoder     ###############
            self.codebook = pretrained_model.codebook    ###############
            self.decoder_gt = pretrained_model.decoder_gt   ###############
            self.decoder_2_gt =   pretrained_model.decoder_2_gt        ###############
            ###############################################


            if self.mode == 'transformer':
                for p in self.parameters():
                    p.requires_grad = False

                transformer_config = {
                    "vocab_size": 256,
                    "block_size": 512,
                    "n_layer": 24,
                    "n_head": 16,
                    "n_embd": 1024
                }
                self.transformer = GPT(**transformer_config)



    def get_state_encoding(self, past, abs_past, seq_start_end, end_pose, future, ground_truth):
        # ***
        # past: 标准化过的过去轨迹（除以1000）
        # abs_past: 没有标准化过的过去轨迹
        # seq_start_end: 每个seqence
        # end_pose: 每条轨迹的最后意图（未来轨迹的最后一帧）
        # future: 未来轨迹的最后两帧
        # ***
        norm_past_state = self.norm_past_encoder(past)   #[513, 64]
        abs_past_state = self.abs_past_encoder(abs_past)   #[513, 64]
        # norm_fut_state = self.norm_fut_encoder(future)    #[513, 64]
        norm_gt_state = self.norm_gt_encoder(ground_truth)    #[513, 96]

        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)

        ##############################################################################################

        norm_past_state_codebook = norm_past_state.view(-1, self.codebook_lantent_dim)
        norm_gt_state_codebook = norm_gt_state.view(-1, self.codebook_lantent_dim)
        abs_past_state_social_codebook = abs_past_state_social.view(-1, self.codebook_lantent_dim)
        codebook_total = torch.cat((norm_past_state_codebook, norm_gt_state_codebook, abs_past_state_social_codebook))

        num_norm_past_state_codebook = norm_past_state_codebook.size(0)
        num_norm_gt_state_codebook = norm_gt_state_codebook.size(0)


        codebook_total_mapping, codebook_total_indices, q_loss = self.codebook(codebook_total)

        norm_past_state_codebook_after = codebook_total_mapping[:num_norm_past_state_codebook, :]
        norm_gt_state_codebook_after = codebook_total_mapping[num_norm_past_state_codebook:num_norm_past_state_codebook+num_norm_gt_state_codebook, :]
        abs_past_state_social_codebook_after = codebook_total_mapping[num_norm_past_state_codebook+num_norm_gt_state_codebook:, :]

        norm_past_state_codebook_after = norm_past_state_codebook_after.view(norm_past_state.size())
        norm_gt_state_codebook_after = norm_gt_state_codebook_after.view(norm_gt_state.size())
        abs_past_state_social_codebook_after = abs_past_state_social_codebook_after.view(abs_past_state_social.size())


        return norm_past_state_codebook_after, abs_past_state_social_codebook_after, norm_gt_state_codebook_after, q_loss


    def decode_state_into_intention(self, past, norm_past_state, abs_past_state_social, ground_truth, norm_gt_state):
        # state concatenation and decoding
        input_fut = torch.cat((norm_past_state, abs_past_state_social), 1)
        # prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)

        diff_past = past - reconstruction_x1 # B, T, 2  [513, 8, 2]
        diff_past_embed = self.res_past_encoder(diff_past) # B, F  [513, 64]

        state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social), 1)
        # prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
        reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)

        ######################
        pred_gt1 = self.decoder_gt(input_fut).contiguous().view(-1, self.gt_len, 2)
        input_gt = torch.cat((diff_past_embed, abs_past_state_social, norm_gt_state), dim=1)
        pred_gt2 = self.decoder_2_gt(input_gt).contiguous().view(-1, self.gt_len, 2)
        pred_gt = pred_gt1 + pred_gt2
        ########################
        
        # prediction = prediction_y1 + prediction_y2
        reconstruction = reconstruction_x1 + reconstruction_x2
        
        return reconstruction, pred_gt

    def reconstruction(self, past, abs_past, seq_start_end, end_pose, future, ground_truth):

        ########################################
        #编码得到原始特征
        norm_past_state = self.norm_past_encoder(past)  # [513, 64]
        abs_past_state = self.abs_past_encoder(abs_past)  # [513, 64]
        norm_gt_state = self.norm_gt_encoder(ground_truth)  # [513, 96]
        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)

        #############################################
        #经过VQ过程，将原始特征映射为CodeBook中的向量

        #1. 将特征拉平并cat
        norm_past_state_codebook = norm_past_state.view(-1, self.codebook_lantent_dim)
        norm_gt_state_codebook = norm_gt_state.view(-1, self.codebook_lantent_dim)
        abs_past_state_social_codebook = abs_past_state_social.view(-1, self.codebook_lantent_dim)
        codebook_total = torch.cat((norm_past_state_codebook, norm_gt_state_codebook, abs_past_state_social_codebook))

        num_norm_past_state_codebook = norm_past_state_codebook.size(0)
        num_norm_gt_state_codebook = norm_gt_state_codebook.size(0)

        #进行映射
        codebook_total_mapping, codebook_total_indices, q_loss = self.codebook(codebook_total)

        norm_past_state_codebook_after = codebook_total_mapping[:num_norm_past_state_codebook, :]
        norm_gt_state_codebook_after = codebook_total_mapping[
                                       num_norm_past_state_codebook:num_norm_past_state_codebook + num_norm_gt_state_codebook,
                                       :]
        abs_past_state_social_codebook_after = codebook_total_mapping[
                                               num_norm_past_state_codebook + num_norm_gt_state_codebook:, :]

        # 分别得到映射后的特征
        norm_past_state_codebook_after = norm_past_state_codebook_after.view(norm_past_state.size())
        norm_gt_state_codebook_after = norm_gt_state_codebook_after.view(norm_gt_state.size())
        abs_past_state_social_codebook_after = abs_past_state_social_codebook_after.view(abs_past_state_social.size())

        # #################################
        # 进行解码重构
        fusion_past_state = norm_past_state + norm_past_state_codebook_after
        fusion_gt_state = norm_gt_state + norm_gt_state_codebook_after

        input_fut = torch.cat((fusion_past_state, abs_past_state_social, fusion_gt_state), 1)
        # prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
        pred_gt1 = self.decoder_gt(input_fut).contiguous().view(-1, self.gt_len, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)

        diff_past = past - reconstruction_x1  # B, T, 2  [513, 8, 2]
        diff_past_embed = self.res_past_encoder(diff_past)  # B, F  [513, 64]

        state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, fusion_gt_state), 1)
        # prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
        pred_gt2 = self.decoder_2_gt(state_conc_diff).contiguous().view(-1, self.gt_len, 2)
        reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)

        # prediction = prediction_y1 + prediction_y2
        pred_gt = pred_gt1 + pred_gt2
        reconstruction = reconstruction_x1 + reconstruction_x2

        return reconstruction, pred_gt, q_loss

    def train_tranformer(self, past, abs_past, seq_start_end, end_pose, future, ground_truth):

        norm_past_state = self.norm_past_encoder(past)  # [513, 64]
        norm_gt_state = self.norm_gt_encoder(ground_truth)  # [513, 96]
        abs_past_state = self.abs_past_encoder(abs_past)  # [513, 64]
        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)

        ##############################################################################################

        norm_past_state_codebook = norm_past_state.view(-1, self.codebook_lantent_dim)
        norm_gt_state_codebook = norm_gt_state.view(-1, self.codebook_lantent_dim)
        codebook_total = torch.cat((norm_past_state_codebook, norm_gt_state_codebook))

        num_norm_past_state_codebook = norm_past_state_codebook.size(0)

        past_mapping, codebook_total_indices, _ = self.codebook(codebook_total)
        past_mapping = past_mapping[:num_norm_past_state_codebook].view(norm_past_state.shape[0], -1)

        norm_past_indices = codebook_total_indices[:num_norm_past_state_codebook].view(norm_gt_state.shape[0], -1)
        norm_gt_state_indices = codebook_total_indices[num_norm_past_state_codebook:].view(
            norm_gt_state.shape[0], -1)

        ##############################################################################################

        target = norm_gt_state_indices

        random_indices = torch.randint_like(norm_gt_state_indices, self.transformer.config.vocab_size)


        new_indices = torch.cat((norm_past_indices, random_indices), dim=1)

        # for k in range(steps):
        #     logits, _ = self.transformer(x)
        #     logits = logits[:, -1, :] / temperature
        #
        #     if top_k is not None:
        #         logits = self.top_k_logits(logits, top_k)
        #
        #     probs = F.softmax(logits, dim=-1)
        #
        #     ix = torch.multinomial(probs, num_samples=1)
        #
        #     x = torch.cat((x, ix), dim=1)

        logits, _ = self.transformer(new_indices)
        probs = F.softmax(logits, -1)
        pred_indices = torch.Tensor().cuda()
        for i in range(probs.shape[1]):
            idx = probs[:, i, :]
            idx = torch.multinomial(idx, num_samples=1)
            pred_indices = torch.cat((pred_indices, idx), dim=1)

        codebook_mapping = self.codebook.embedding(pred_indices[:, 4:].long()).view(logits.shape[0], -1)
        codebook_mapping = codebook_mapping.view(norm_gt_state.shape[0], -1)

        fusion_past_state = norm_past_state + past_mapping
        fusion_gt_state = norm_gt_state + codebook_mapping

        input_fut = torch.cat((fusion_past_state, abs_past_state_social, fusion_gt_state), 1)
        # prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
        pred_gt1 = self.decoder_gt(input_fut).contiguous().view(-1, self.gt_len, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)

        diff_past = past - reconstruction_x1  # B, T, 2  [513, 8, 2]
        diff_past_embed = self.res_past_encoder(diff_past)  # B, F  [513, 64]

        state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, fusion_gt_state), 1)
        # prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
        pred_gt2 = self.decoder_2_gt(state_conc_diff).contiguous().view(-1, self.gt_len, 2)
        reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)

        # prediction = prediction_y1 + prediction_y2
        pred_gt = pred_gt1 + pred_gt2
        reconstruction = reconstruction_x1 + reconstruction_x2

        # reconstruction, pred_gt = self.decode_state_into_intention(past, norm_past_state, abs_past_state_social, ground_truth, codebook_mapping)


        return logits[:, 4:, :], target, pred_gt



    def forward(self, past, abs_past, seq_start_end, end_pose, future, ground_truth):

        if self.mode == 'intention':
            # norm_past_state, abs_past_state_social, norm_gt_state, q_loss, _, _, _ = \
            #     self.get_state_encoding(past, abs_past, seq_start_end, end_pose, future, ground_truth)
            # reconstruction, pred_gt = self.decode_state_into_intention(
            #     past, norm_past_state, abs_past_state_social, ground_truth, norm_gt_state)
            reconstruction, pred_gt, q_loss = self.reconstruction(past, abs_past, seq_start_end, end_pose, future, ground_truth)
            return reconstruction, pred_gt, q_loss
        elif self.mode == 'transformer':
            # norm_past_state, abs_past_state_social, norm_gt_state, q_loss, _, _, _ = \
            #     self.get_state_encoding(past, abs_past, seq_start_end, end_pose, future, ground_truth)
            logits, target, pred_gt = self.train_tranformer(past, abs_past, seq_start_end, end_pose, future, ground_truth)
            # self.train_transformer(norm_past_state, abs_past_state_social, norm_gt_state)
            return logits, target, pred_gt
