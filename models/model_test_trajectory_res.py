import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mingpt import GPT


class model_encdec(nn.Module):
    
    def __init__(self, settings, pretrained_model):
        super(model_encdec, self).__init__()

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = pretrained_model.dim_embedding_key
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.gt_len = pretrained_model.gt_len
        self.codebook_lantent_dim = pretrained_model.codebook_lantent_dim
        
        # LAYERS
        self.abs_past_encoder = pretrained_model.abs_past_encoder
        self.norm_past_encoder = pretrained_model.norm_past_encoder
        self.norm_fut_encoder = pretrained_model.norm_fut_encoder
        self.res_past_encoder = pretrained_model.res_past_encoder
        self.social_pooling_X = pretrained_model.social_pooling_X
        # self.decoder = pretrained_model.decoder
        self.decoder_x = pretrained_model.decoder_x
        # self.decoder_x_abs = pretrained_model.decoder_x_abs
        # self.decoder_2 = pretrained_model.decoder_2
        self.decoder_2_x = pretrained_model.decoder_2_x

        self.norm_gt_encoder = pretrained_model.norm_gt_encoder  ###############
        self.res_gt_encoder = pretrained_model.res_gt_encoder  ###############
        self.codebook = pretrained_model.codebook  ###############
        self.decoder_gt = pretrained_model.decoder_gt  ###############
        self.decoder_2_gt = pretrained_model.decoder_2_gt  ###############
        self.transformer = pretrained_model.transformer    ###############

        # self.decoder_2_x_abs = pretrained_model.decoder_2_x_abs
        # self.input_query_w = pretrained_model.input_query_w
        # self.past_memory_w = pretrained_model.past_memory_w

        # MEMORY
        # self.memory_past = torch.load('./training/saved_memory/sdd_social_0.5_0.5_15442_filter_past.pt').cuda()
        # self.memory_fut = torch.load('./training/saved_memory/sdd_social_0.5_0.5_15442_filter_fut.pt').cuda()
        
        # self.encoder_dest = pretrained_model.encoder_dest
        # self.encoder_dest = MLP(input_dim = 2, output_dim = 64, hidden_size=(64, 128))
        # self.traj_abs_past_encoder = pretrained_model.traj_abs_past_encoder
        # self.interaction = pretrained_model.interaction
        # self.num_decompose = 2
        # self.decompose = pretrained_model.decompose
        


        # activation function

        for p in self.parameters():
            p.requires_grad = False

    def get_state_encoding(self, past, abs_past, seq_start_end, end_pose, future):
        norm_past_state = self.norm_past_encoder(past)
        abs_past_state = self.abs_past_encoder(abs_past)
        norm_fut_state = self.norm_fut_encoder(future)


        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)
        
        return norm_past_state, abs_past_state_social, norm_fut_state


    @torch.no_grad()
    def sample(self, past, steps, temperature=1.0, top_k=20):
        self.transformer.eval()
        # x = torch.cat((c, past), dim=1)
        x = past
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        # x = x[:, past.shape[1]:]
        self.transformer.train()
        return x

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    def decode_state_into_intention(self, past, norm_past_state, abs_past_state_social, ground_truth, norm_gt_state):
        # state concatenation and decoding
        input_fut = torch.cat((norm_past_state, abs_past_state_social), 1)
        # prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
        reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)

        diff_past = past - reconstruction_x1  # B, T, 2  [513, 8, 2]
        diff_past_embed = self.res_past_encoder(diff_past)  # B, F  [513, 64]

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

    def forward(self, past, abs_past, seq_start_end, end_pose, ground_truth):
        prediction = torch.Tensor().cuda()

        norm_past_state = self.norm_past_encoder(past)  # [513, 64]
        abs_past_state = self.abs_past_encoder(abs_past)  # [513, 64]
        # norm_fut_state = self.norm_fut_encoder(future)    #[513, 64]
        norm_gt_state = self.norm_gt_encoder(ground_truth)  # [513, 96]

        abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)

        ##############################################################################################

        norm_past_state_codebook = norm_past_state.view(-1, self.codebook_lantent_dim)
        norm_gt_state_codebook = norm_gt_state.view(-1, self.codebook_lantent_dim)
        codebook_total = torch.cat((norm_past_state_codebook, norm_gt_state_codebook))

        num_norm_past_state_codebook = norm_past_state_codebook.size(0)

        _, codebook_total_indices, _ = self.codebook(codebook_total)

        norm_past_indices = codebook_total_indices[:num_norm_past_state_codebook].view(norm_gt_state.shape[0], -1)
        norm_gt_state_indices = codebook_total_indices[num_norm_past_state_codebook:].view(
            norm_gt_state.shape[0], -1)


        mask = torch.randn(norm_past_state.shape[0], 6)
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(mask, self.transformer.config.vocab_size).to(device=norm_past_state.device)

        new_indices = torch.cat((norm_past_indices, random_indices), dim=1)

        # logits = self.sample(norm_past_indices, random_indices, 6)

        logits, _ = self.transformer(new_indices)

        loss = F.cross_entropy(logits[:, 4:, :].reshape(-1, logits.size(-1)), norm_gt_state_indices.reshape(-1))

        probs = F.softmax(logits, -1)
        pred_indices = torch.Tensor().cuda()
        for i in range(probs.shape[1]):
            idx = probs[:, i, :]
            idx = torch.multinomial(idx, num_samples=1)
            pred_indices = torch.cat((pred_indices, idx), dim=1)

        pred_mapping = self.codebook.embedding(pred_indices[:, 4:].long())
        pred_mapping = pred_mapping.view(pred_mapping.shape[0], -1)

        input_gt = torch.cat((norm_past_state, abs_past_state_social, pred_mapping), dim=1)
        pred_gt1 = self.decoder_gt(input_gt).contiguous().view(-1, self.gt_len, 2)
        reconstruction_x1 = self.decoder_x(input_gt).contiguous().view(-1, self.past_len, 2)

        diff_gt = past - reconstruction_x1
        # diff_gt = pred_gt1
        diff_gt_embed = self.res_gt_encoder(diff_gt)
        state_input_gt = torch.cat((diff_gt_embed, abs_past_state_social, pred_mapping), dim=1)
        pred_gt2 = self.decoder_2_gt(state_input_gt).contiguous().view(-1, self.gt_len, 2)
        reconstruction_x2 = self.decoder_2_x(state_input_gt).contiguous().view(-1, self.past_len, 2)

        prediction = pred_gt1 + pred_gt2
        reconstruction = reconstruction_x1 +reconstruction_x2


        # B, K, T, 2
        return prediction, loss