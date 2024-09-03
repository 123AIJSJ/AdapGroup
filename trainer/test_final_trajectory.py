import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_test_trajectory_res import model_encdec
from trainer.evaluations import evaluate_transformer_test

from sddloader import *

torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'testing/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
 
       
        self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)

       
        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

        self.settings = {
            "train_batch_size": config.train_b_size,
            "test_batch_size": config.test_b_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 12,
        }

        # model
        self.model_ae = torch.load(config.model_ae).cuda()
        self.mem_n2n = model_encdec(self.settings, self.model_ae)
        

        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

        


    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
        return 0

    
    def fit(self):
        
        dict_metrics_test, loss = evaluate_transformer_test(self.test_dataset, self.mem_n2n, self.config, self.device)
        print('Test loss: {} ------  ade:: {}'.format(dict_metrics_test, loss))
        print('-'*100)
        

    def evaluate(self, dataset):
        
        ade_48s = fde_48s = 0
        samples = 0
        dict_metrics = {}
        loss_all = 0
        count = 0

        with torch.no_grad():
            for i, (traj, mask, initial_pos,seq_start_end) \
                in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
                traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)
                # traj (B, T, 2)
                initial_pose = traj[:, 7, :] / 1000
                
                traj_norm = traj - traj[:, 7:8, :]
                x = traj_norm[:, :self.config.past_len, :]
                destination = traj_norm[:, -2:, :]
                ground_truth = traj_norm[:, 8:, :]
                

                abs_past = traj[:, :self.config.past_len, :]
                
                output, loss = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, ground_truth)
                output = output.data
                loss_all += loss
                # B, K, t, 2

                # future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
                distances = torch.norm(output - traj_norm[:, 8:, :], dim=2)
                mean_distances = torch.mean(distances[:, -1:], dim=1)
                # index_min = torch.argmin(mean_distances, dim=1)
                # min_distances = distances[torch.arange(0, len(index_min)), index_min]

                # fde_48s += torch.sum(min_distances[:, -1])
                fde_48s += torch.sum(mean_distances)
                ade_48s += torch.sum(torch.mean(distances, dim=1))
                samples += distances.shape[0]
                count += 1


            dict_metrics['fde_48s'] = fde_48s / samples
            dict_metrics['ade_48s'] = ade_48s / samples

        return dict_metrics, loss_all / count

    def evaluate_2(self, dataset, model, config, device):
        ade = 0
        loss = 0
        samples = 0

        with torch.no_grad():
            for i, (traj, mask, initial_pos, seq_start_end) \
                    in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches,
                                     dataset.seq_start_end_batches)):
                traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), torch.FloatTensor(mask).to(
                    self.device), torch.FloatTensor(initial_pos).to(self.device)
                # traj (B, T, 2)
                initial_pose = traj[:, 7, :] / 1000

                traj_norm = traj - traj[:, 7:8, :]
                x = traj_norm[:, :self.config.past_len, :]
                destination = traj_norm[:, -2:, :]
                ground_truth = traj_norm[:, 8:, :]

                abs_past = traj[:, :self.config.past_len, :]

                norm_past_state = self.mem_n2n.norm_past_encoder(x)  # [513, 64]
                norm_gt_state = self.mem_n2n.norm_gt_encoder(ground_truth)  # [513, 96]
                abs_past_state = self.mem_n2n.abs_past_encoder(abs_past)  # [513, 64]
                abs_past_state_social = self.mem_n2n.social_pooling_X(abs_past_state, seq_start_end, initial_pose)

                ##############################################################################################

                norm_past_state_codebook = norm_past_state.view(-1, self.mem_n2n.codebook_lantent_dim)
                norm_gt_state_codebook = norm_gt_state.view(-1, self.mem_n2n.codebook_lantent_dim)
                codebook_total = torch.cat((norm_past_state_codebook, norm_gt_state_codebook))

                num_norm_past_state_codebook = norm_past_state_codebook.size(0)

                _, codebook_total_indices, _ = self.mem_n2n.codebook(codebook_total)

                norm_past_indices = codebook_total_indices[:num_norm_past_state_codebook].view(norm_gt_state.shape[0],
                                                                                               -1)
                norm_gt_state_indices = codebook_total_indices[num_norm_past_state_codebook:].view(
                    norm_gt_state.shape[0], -1)

                ##############################################################################################

                targets = norm_gt_state_indices

                random_indices = torch.randint_like(norm_gt_state_indices, self.mem_n2n.transformer.config.vocab_size)

                fusion_indices = torch.cat((norm_past_indices, random_indices), dim=1)

                logits = self.mem_n2n.sample(norm_past_indices, steps=6)

                # logits, _ = self.mem_n2n.transformer(fusion_indices)
                # probs = F.softmax(logits, -1)
                # pred_indices = torch.Tensor().cuda()
                # for i in range(probs.shape[1]):
                #     idx = probs[:, i, :]
                #     idx = torch.multinomial(idx, num_samples=1)
                #     pred_indices = torch.cat((pred_indices, idx), dim=1)

                codebook_mapping = self.mem_n2n.codebook.embedding(logits[:, 4:].long()).view(logits.shape[0], -1)

                reconstruction, pred_gt = self.mem_n2n.decode_state_into_intention(x, norm_past_state, abs_past_state_social,
                                                                            ground_truth, codebook_mapping)
                ade += self.get_ade(pred_gt, ground_truth)
                # a =
                # b = torch.mean(a, )
                loss += torch.mean(torch.mean((logits[:, 4:] - norm_gt_state_indices).float(), dim=1))
                samples += 1

        return loss / samples, ade / samples

    def get_ade(self, pred, target):

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += torch.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all = sum_ / (N * T)

        return sum_all
