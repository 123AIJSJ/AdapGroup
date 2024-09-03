import torch
import torch.nn.functional as F



###计算ade，采样时只计算一次
def evaluate_transformer(dataset, model, config, device):
    loss = 0
    samples = 0
    ade_48s = fde_48s = 0
    dict_metrics = {}

    with torch.no_grad():
        for i, (traj, mask, initial_pos, seq_start_end) \
                in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches,
                                 dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(
                device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len - 1, :] / 1000

            traj_norm = traj - traj[:, config.past_len - 1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]
            y = traj_norm[:, -1:, :]
            past = x
            end_pose = initial_pose

            ground_truth = traj_norm[:, config.past_len:, :]

            abs_past = traj[:, :config.past_len, :]

            # logits, targets = model(x, abs_past, seq_start_end, initial_pose, destination, ground_truth)
            #
            # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            prediction = torch.Tensor().cuda()


            norm_past_state = model.norm_past_encoder(past)  # [513, 64]
            abs_past_state = model.abs_past_encoder(abs_past)  # [513, 64]

            abs_past_state_social = model.social_pooling_X(abs_past_state, seq_start_end, end_pose)

            ##############################################################################################

            norm_past_state_codebook = norm_past_state.view(-1, model.codebook_lantent_dim)

            codebook_total_mapping, codebook_total_indices, q_loss = model.codebook(norm_past_state_codebook)

            norm_past_indices = codebook_total_indices.view(norm_past_state.shape[0], -1)

            mask = torch.randn(norm_past_state.shape[0], 6)
            mask = mask.round().to(dtype=torch.int64)
            random_indices = torch.randint_like(mask, model.transformer.config.vocab_size).to(
                device=norm_past_state.device)

            new_indices = torch.cat((norm_past_indices, random_indices), dim=1)

            logits, _ = model.transformer(new_indices)

            probs = F.softmax(logits, -1)
            pred_indices = torch.Tensor().cuda()
            for i in range(probs.shape[1]):
                idx = probs[:, i, :]
                idx = torch.multinomial(idx, num_samples=1)
                pred_indices = torch.cat((pred_indices, idx), dim=1)

            pred_mapping = model.codebook.embedding(pred_indices[:, 4:].long())
            pred_mapping = pred_mapping.view(pred_mapping.shape[0], -1)

            input_gt = torch.cat((norm_past_state, abs_past_state_social, pred_mapping), dim=1)
            pred_gt1 = model.decoder_gt(input_gt).contiguous().view(-1, model.gt_len, 2)

            # diff_gt = ground_truth - pred_gt1
            diff_gt = pred_gt1
            diff_gt_embed = model.res_gt_encoder(diff_gt)
            state_input_gt = torch.cat((norm_past_state, abs_past_state_social, diff_gt_embed), dim=1)
            pred_gt2 = model.decoder_2_gt(state_input_gt).contiguous().view(-1, model.gt_len, 2)
            prediction = pred_gt1 + pred_gt2

            output = prediction.data
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

        dict_metrics['fde_48s'] = fde_48s / samples
        dict_metrics['ade_48s'] = ade_48s / samples

    return dict_metrics


##和训练transformer一样
def evaluate_transformer_t(dataset, model, config, device):
    loss = 0
    ade = 0
    samples = 0


    with torch.no_grad():
        for i, (traj, mask, initial_pos, seq_start_end) \
                in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches,
                                 dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(
                device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len - 1, :] / 1000

            traj_norm = traj - traj[:, config.past_len - 1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]

            ground_truth = traj_norm[:, config.past_len:, :]

            abs_past = traj[:, :config.past_len, :]

            logits, targets, pred_gt = model(x, abs_past, seq_start_end, initial_pose, destination, ground_truth)

            loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            ade += get_ade(pred_gt, ground_truth)
            samples += 1

    return loss / samples, ade / samples

def get_ade(pred, target):

    N = pred.shape[0]
    T = pred.shape[1]
    sum_ = 0
    for i in range(N):
        for t in range(T):
            sum_ += torch.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
    sum_all = sum_ / (N * T)

    return sum_all


###   和测试时一样，计算ade，采样6次
def evaluate_transformer_test(dataset, model, config, device):
    ade = 0
    samples = 0
    loss = 0

    with torch.no_grad():
        for i, (traj, mask, initial_pos, seq_start_end) \
                in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches,
                                 dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(
                device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len - 1, :] / 1000

            traj_norm = traj - traj[:, config.past_len - 1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]

            ground_truth = traj_norm[:, config.past_len:, :]

            abs_past = traj[:, :config.past_len, :]

            norm_past_state = model.norm_past_encoder(x)  # [513, 64]
            norm_gt_state = model.norm_gt_encoder(ground_truth)  # [513, 96]
            abs_past_state = model.abs_past_encoder(abs_past)  # [513, 64]
            abs_past_state_social = model.social_pooling_X(abs_past_state, seq_start_end, initial_pose)

            ##############################################################################################

            norm_past_state_codebook = norm_past_state.view(-1, model.codebook_lantent_dim)
            norm_gt_state_codebook = norm_gt_state.view(-1, model.codebook_lantent_dim)
            codebook_total = torch.cat((norm_past_state_codebook, norm_gt_state_codebook))

            num_norm_past_state_codebook = norm_past_state_codebook.size(0)

            past_mapping, codebook_total_indices, _ = model.codebook(codebook_total)
            past_mapping = past_mapping[:num_norm_past_state_codebook].view(norm_past_state.shape[0], -1)

            norm_past_indices = codebook_total_indices[:num_norm_past_state_codebook].view(norm_gt_state.shape[0], -1)
            norm_gt_state_indices = codebook_total_indices[num_norm_past_state_codebook:].view(
                norm_gt_state.shape[0], -1)

            ##############################################################################################

            targets = norm_gt_state_indices

            random_indices = torch.randint_like(norm_gt_state_indices, model.transformer.config.vocab_size)

            fusion_indices = torch.cat((norm_past_indices, random_indices), dim=1)

            logits, _ = model.transformer(fusion_indices)
            probs = F.softmax(logits, -1)
            pred_indices = torch.Tensor().cuda()
            for i in range(probs.shape[1]):
                idx = probs[:, i, :]
                idx = torch.multinomial(idx, num_samples=1)
                pred_indices = torch.cat((pred_indices, idx), dim=1)

            codebook_mapping = model.codebook.embedding(pred_indices[:, 4:].long()).view(logits.shape[0], -1)
            codebook_mapping = codebook_mapping.view(norm_gt_state.shape[0], -1)

            fusion_past_state = norm_past_state + past_mapping
            fusion_gt_state = norm_gt_state + codebook_mapping

            input_fut = torch.cat((fusion_past_state, abs_past_state_social, fusion_gt_state), 1)
            # prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
            pred_gt1 = model.decoder_gt(input_fut).contiguous().view(-1, model.gt_len, 2)
            reconstruction_x1 = model.decoder_x(input_fut).contiguous().view(-1, model.past_len, 2)

            diff_past = x - reconstruction_x1  # B, T, 2  [513, 8, 2]
            diff_past_embed = model.res_past_encoder(diff_past)  # B, F  [513, 64]

            state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, fusion_gt_state), 1)
            # prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
            pred_gt2 = model.decoder_2_gt(state_conc_diff).contiguous().view(-1, model.gt_len, 2)
            reconstruction_x2 = model.decoder_2_x(state_conc_diff).contiguous().view(-1, model.past_len, 2)

            # prediction = prediction_y1 + prediction_y2
            pred_gt = pred_gt1 + pred_gt2
            reconstruction = reconstruction_x1 + reconstruction_x2

            # reconstruction, pred_gt = model.decode_state_into_intention(x, norm_past_state, abs_past_state_social, ground_truth, codebook_mapping)
            ade += get_ade(pred_gt, ground_truth)
            current_loss = F.cross_entropy(logits[:, 4:, :].reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss += current_loss
            samples += 1

    return loss / samples, ade / samples


def evaluate_intention(dataset, model, config, device): 
    ade_pred = 0
    ade_past = 0
    count = 0

    with torch.no_grad():
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len-1, :] / 1000
            
            traj_norm = traj - traj[:, config.past_len-1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]
            y = traj_norm[:, -1:, :]

            ground_truth = traj_norm[:, config.past_len:, :]

            abs_past = traj[:, :config.past_len, :]

            recon, pred_gt, q_loss = model(x, abs_past, seq_start_end, initial_pose, destination, ground_truth)
            # output = output.data
            recon = recon.data
            pred_gt = pred_gt.data

            ade_past += get_ade(recon, x)
            ade_pred += get_ade(pred_gt, ground_truth)
            count += 1

    return ade_past / count, ade_pred / count

def get_ade(pred, target):

    N = pred.shape[0]
    T = pred.shape[1]
    sum_ = 0
    for i in range(N):
        for t in range(T):
            sum_ += torch.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
    sum_all = sum_ / (N * T)

    return sum_all



def evaluate_addressor(train_dataset, dataset, model, config, device): 
    fde_48s = 0
    samples = 0

    model.generate_memory(train_dataset, filter_memory=False)

    with torch.no_grad():
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len-1, :] / 1000
            
            traj_norm = traj - traj[:, config.past_len-1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            y = traj_norm[:, -1:, :]

            abs_past = traj[:, :config.past_len, :]

            output = model.get_destination_from_memory(x, abs_past, seq_start_end, initial_pose)
            output = output.data

            future_rep = y.unsqueeze(1).repeat(1, 20, 1, 1)
            distances = torch.norm(output - future_rep, dim=3)
            mean_distances = torch.mean(distances[:, :, -1:], dim=2)
            index_min = torch.argmin(mean_distances, dim=1)
            min_distances = distances[torch.arange(0, len(index_min)), index_min]

            fde_48s += torch.sum(min_distances[:, -1])
            samples += distances.shape[0]

    return fde_48s / samples


def evaluate_trajectory(dataset, model, config, device):
    # for the fulfillment stage or trajectory stage, we should have a fixed past/intention memory bank.
    ade_48s = fde_48s = 0
    samples = 0
    dict_metrics = {}

    with torch.no_grad():
        for i, (traj, mask, initial_pos,seq_start_end) \
            in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches, dataset.seq_start_end_batches)):
            traj, mask, initial_pos = torch.FloatTensor(traj).to(device), torch.FloatTensor(mask).to(device), torch.FloatTensor(initial_pos).to(device)
            # traj (B, T, 2)
            initial_pose = traj[:, config.past_len-1, :] / 1000
            
            traj_norm = traj - traj[:, config.past_len-1:config.past_len, :]
            x = traj_norm[:, :config.past_len, :]
            destination = traj_norm[:, -2:, :]

            abs_past = traj[:, :config.past_len, :]

            output = model.get_trajectory(x, abs_past, seq_start_end, initial_pose)
            output = output.data

            future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
            distances = torch.norm(output - future_rep, dim=3)
            mean_distances = torch.mean(distances[:, :, -1:], dim=2) # find the tarjectory according to the last frame's distance
            index_min = torch.argmin(mean_distances, dim=1)
            min_distances = distances[torch.arange(0, len(index_min)), index_min]

            fde_48s += torch.sum(min_distances[:, -1])
            ade_48s += torch.sum(torch.mean(min_distances, dim=1))
            samples += distances.shape[0]


        dict_metrics['fde_48s'] = fde_48s / samples
        dict_metrics['ade_48s'] = ade_48s / samples

    return dict_metrics['ade_48s']
