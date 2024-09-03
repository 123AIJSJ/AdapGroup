import os
import datetime
import torch
import torch.nn as nn
from models.model_AIO import model_encdec
from trainer.evaluations import *
from models.layer_utils import print_log
import time

from sddloader import *


torch.set_num_threads(5)

class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'training/training_' + config.mode + '/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.log = open(os.path.join(config.log), 'a+')
 
        # print('Creating dataset...')
        if config.mode != 'test':
            self.train_dataset = SocialDataset(set_name="train", b_size=config.train_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)
        self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh, d_tresh=config.dist_thresh)

        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

        self.settings = {
            "mode": config.mode,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 1 if config.mode=='intention' else 12,
            "gt_len": 12,
            "num_codebook_vectors": config.num_codebook_vectors,
        }
        self.max_epochs = config.max_epochs

        # model
        if config.mode == 'intention':
            self.mem_n2n = model_encdec(self.settings)
        else:
            self.model_ae = torch.load(config.model_ae).cuda()
            self.mem_n2n = model_encdec(self.settings, self.model_ae)
            print_log("Mode: {}\n".format(config.mode), self.log)

        # optimizer and learning rate
        if config.mode == 'transformer':
            self.criterionLoss = nn.CrossEntropyLoss()
        else:
            self.criterionLoss = nn.MSELoss()
        if config.mode == 'addressor':
            config.learning_rate = 1e-6

        # trainable_layers = self.mem_n2n.get_parameters(config.mode)
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
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
        self.print_model_param(self.mem_n2n)
        print_log('\n----------\nDataset: {}\nMode: {}\n----------\n'.format("sdd", self.config.mode), self.log)
        minValue = 100
        for epoch in range(self.start_epoch, self.config.max_epochs):

            # print(' ----- Epoch: {}'.format(epoch))
            # print('Loss: {}'.format(loss))
            loss, ade = self._train_single_epoch()
            if self.config.mode == 'intention':
                print_log('[{}] Epoch: {}/{}\tMSE_Loss: {:.6f}   -----future_ade:: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                                   str(epoch), self.config.max_epochs, loss, ade), log=self.log)
            else:
                print_log('[{}] Epoch: {}/{}\tindices_loss: {:.6f}   -----pred_loss:: {}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    str(epoch), self.config.max_epochs, loss, ade), log=self.log)

            # if (epoch + 1) % 1 == 0:
            if True:
                if self.config.mode == 'intention':
                    currentValue, ade = evaluate_intention(self.test_dataset, self.mem_n2n, self.config, self.device)
                elif self.config.mode == 'transformer':
                    currentValue, ade = evaluate_transformer_test(self.test_dataset, self.mem_n2n, self.config, self.device)

                if ade < minValue:
                    minValue = ade
                    # print('min value: {}'.format(minValue))
                    if self.config.mode == 'intention':
                        print_log('------ past_ade:: {} ------future_ade:: {}'.format(currentValue, minValue), log=self.log)
                    else:
                        print_log('------ indices_loss:: {} ------pred_loss:: {}'.format(currentValue, minValue), log=self.log)
                    # print_log('------ min value:: {} ------gt_loss:: {}'.format(minValue, b), log=self.log)
                    torch.save(self.mem_n2n, self.folder_test + 'model_ae_' + self.name_test)


    def AttentionLoss(self, sim, distance):
        dis_mask = nn.MSELoss(reduction='sum')
        threshold_distance = 80
        mask = torch.where(distance>threshold_distance, torch.zeros_like(distance), torch.ones_like(distance))
        label_sim = (threshold_distance - distance) / threshold_distance
        label_sim = torch.where(label_sim<0, torch.zeros_like(label_sim), label_sim)
        a = dis_mask(sim*mask, label_sim*mask)
        b = (mask.sum()+1e-5)
        loss = a / b
        return loss


    def _train_single_epoch(self):
        
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(self.train_dataset.trajectory_batches, self.train_dataset.mask_batches,
                             self.train_dataset.initial_pos_batches, self.train_dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)
            
            initial_pose = traj[:, self.config.past_len-1, :] / 1000    # 意图 512  2
            
            traj_norm = traj - traj[:, self.config.past_len-1:self.config.past_len, :]  #减去观测轨迹的最后一帧
            x = traj_norm[:, :self.config.past_len, :]  # 观测轨迹
            destination = traj_norm[:, -2:, :]   # 标准化后的意图
            y = traj_norm[:, self.config.past_len:, :]   #gt

            abs_past = traj[:, :self.config.past_len, :]   #原始观测轨迹

            ground_truth = y

            self.opt.zero_grad()
            if self.config.mode == 'intention':
                y = destination[:, -1:, :]
                recon, pred_gt, q_loss = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination, ground_truth)
                loss = self.criterionLoss(recon, x) + self.criterionLoss(pred_gt, ground_truth) + q_loss
                ade = get_ade(pred_gt, ground_truth).cpu().detach().numpy()
            elif self.config.mode == 'transformer':
                logits, targets, pred_gt = self.mem_n2n(x, abs_past, seq_start_end, initial_pose, destination, ground_truth)
                loss = self.criterionLoss(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                ade = self.get_ade(pred_gt, ground_truth).cpu().detach().numpy()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()
        return loss.item(), ade


    def get_ade(self, pred, target):

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += torch.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
        sum_all = sum_ / (N * T)

        return sum_all