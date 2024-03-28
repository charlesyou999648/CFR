import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F


class TSC(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=1024, K=65536, m=0.999, T=0.07, mlp=False, num_positive=0, targeted=False, tr=1,
                 sep_t=False, tw=1, number=0):
        """
        dim: feature dimension (default: 1024)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(TSC, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.num_positive = num_positive
        self.n_cls = 2
        self.targeted = targeted
        self.tr = tr
        self.sep_t = sep_t
        self.tw = tw
        optimal_target = np.load('optimal_{}_{}_{}.npy'.format(self.n_cls, dim, number))
        optimal_target_order = np.arange(self.n_cls)
        target_repeat = tr * np.ones(self.n_cls)

        optimal_target = torch.Tensor(optimal_target).float()
        target_repeat = torch.Tensor(target_repeat).long()
        optimal_target = torch.cat(
            [optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], dim=0)

        target_labels = torch.cat(
            [torch.Tensor([optimal_target_order[i]]).repeat(target_repeat[i]) for i in range(len(target_repeat))],
            dim=0).long().unsqueeze(-1)

        self.register_buffer("optimal_target", optimal_target)
        self.register_buffer("optimal_target_unique", optimal_target[::self.tr, :].contiguous().transpose(0, 1))
        self.register_buffer("target_labels", target_labels)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_labels", -torch.ones(1, K).long())
        self.register_buffer("class_centroid", torch.randn(self.n_cls, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.class_centroid = nn.functional.normalize(self.class_centroid, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_labels[:, ptr:ptr + batch_size] = labels.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, feat_s_raw, feat_t_raw, im_labels, cos_weight=0.0):
        """
        Input:
            feat_s_raw: a batch of features 1
            feat_t_raw: a batch of features 2
        Output:
            logits, targets
        """
        # compute logits
        # Einstein sum is more intuitive
        # positive logits from augmentation: Nx1
        q = F.normalize(feat_s_raw.float(), dim=-1)
        k = F.normalize(feat_t_raw.float(), dim=-1)
        im_labels = im_labels.cuda()
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if self.targeted:
            queue_negatives = self.queue.clone().detach()
            target_negatives = self.optimal_target.transpose(0, 1)
            l_neg = torch.einsum('nc,ck->nk', [q, torch.cat([queue_negatives, target_negatives], dim=1)])
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # positive logits from queue
        im_labels = im_labels.contiguous().view(-1, 1)

        if self.targeted:
            queue_labels = self.queue_labels.clone().detach()
            target_labels = self.target_labels.transpose(0, 1)

            # compute the optimal matching that minimize moving distance between memory bank anchors and targets
            with torch.no_grad():
                mask = torch.eq(im_labels, torch.cat([queue_labels, torch.full_like(target_labels, -1).to(queue_labels.device)], dim=1)).float()
                # im_labels_all = concat_all_gather(im_labels)
                # features_all = concat_all_gather(q.detach())
                
                im_labels_all = im_labels
                features_all = q.detach()
                
                # update memory bank class centroids
                for one_label in torch.unique(im_labels_all):
                    class_centroid_batch = F.normalize(torch.mean(features_all[im_labels_all[:, 0].eq(one_label), :], dim=0), dim=0)
                    self.class_centroid[one_label] = 0.9*self.class_centroid[one_label] + 0.1*class_centroid_batch
                    self.class_centroid[one_label] = F.normalize(self.class_centroid[one_label], dim=0)

                centroid_target_dist = torch.einsum('nc,ck->nk', [self.class_centroid, self.optimal_target_unique])
                centroid_target_dist = centroid_target_dist.detach().cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(-centroid_target_dist)

                for one_label, one_idx in zip(row_ind, col_ind):
                    if one_label not in im_labels:
                        continue
                    one_indices = torch.Tensor([i+one_idx*self.tr for i in range(self.tr)]).long()
                    tmp = mask[im_labels[:, 0].eq(one_label), :]
                    tmp[:, queue_labels.size(1)+one_indices] = 1
                    mask[im_labels[:, 0].eq(one_label), :] = tmp

            # separate samples and target
            if self.sep_t:
                mask_target = mask.clone()
                mask_target[:, :queue_labels.size(1)] = 0
                mask[:, queue_labels.size(1):] = 0
        else:
            mask = torch.eq(im_labels, self.queue_labels.clone().detach()).float()
        mask_pos_view = torch.zeros_like(mask)

        # sample num_positive from each class
        if self.num_positive > 0:
            for i in range(self.num_positive):
                all_pos_idxs = mask.view(-1).nonzero().view(-1)
                num_pos_per_anchor = mask.sum(1)
                num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
                num_pos_cum[0] = 0
                rand = torch.rand(mask.size(0), device=mask.device)
                idxs = ((rand * num_pos_per_anchor).floor() + num_pos_cum).long()
                idxs = idxs[num_pos_per_anchor.nonzero().view(-1)]
                sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
                mask_pos_view.view(-1)[sampled_pos_idxs] = 1
                mask.view(-1)[sampled_pos_idxs] = 0
        else:
            mask_pos_view = mask.clone()

        if self.targeted and self.sep_t:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_target.clone()
            mask_pos_view += mask_target
        else:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_pos_view.clone()
            mask_pos_view_class[:, self.queue_labels.size(1):] = 0
            mask_pos_view_target[:, :self.queue_labels.size(1)] = 0

        mask_pos_view = torch.cat([torch.ones([mask_pos_view.shape[0], 1]).cuda(), mask_pos_view], dim=1)
        mask_pos_view_class = torch.cat([torch.ones([mask_pos_view_class.shape[0], 1]).cuda(), mask_pos_view_class], dim=1)
        mask_pos_view_target = torch.cat([torch.zeros([mask_pos_view_target.shape[0], 1]).cuda(), mask_pos_view_target], dim=1)

        # apply temperature
        logits /= self.T

        log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
        
        loss_class = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        loss_target = - torch.sum((mask_pos_view_target * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        loss_target = loss_target * self.tw
        loss_logit = F.cross_entropy(logits, labels)
        loss = loss_class + loss_target + loss_logit
        
        if cos_weight>0:
        
            loss_class = - torch.sum((mask_pos_view_class * logits).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0] \
                        + torch.sum(((1 - mask_pos_view_class) * logits).sum(1) / (1-mask_pos_view).sum(1)) / (1-mask_pos_view).shape[0]
            loss_target = - torch.sum((mask_pos_view_target * logits).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0] \
                        + torch.sum(((1 - mask_pos_view_target) * logits).sum(1) / (1-mask_pos_view).sum(1)) / (1-mask_pos_view).shape[0]
            loss_target = loss_target * self.tw
            loss = loss * (1-cos_weight) + cos_weight*(loss_class + loss_target + loss_logit)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, im_labels)
        
        # if sep_pos_neg:
        #     pos_loss = (log_prob * mask_pos_view_class).mean(dim=-1)+(log_prob * mask_pos_view_target).mean(dim=-1)* self.tw
        #     neg_loss = (log_prob * (1- mask_pos_view_class)).mean(dim=-1)+(log_prob * (1- mask_pos_view_target)).mean(dim=-1)* self.tw
        #     return (pos_loss, neg_loss)
        
            
        return loss
    
    
class TSC_cossim(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=1024, K=65536, m=0.999, T=0.07, mlp=False, num_positive=0, targeted=False, tr=1,
                 sep_t=False, tw=1, number=0):
        """
        dim: feature dimension (default: 1024)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(TSC_cossim, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.num_positive = num_positive
        self.n_cls = 2
        self.targeted = targeted
        self.tr = tr
        self.sep_t = sep_t
        self.tw = tw
        optimal_target = np.load('optimal_{}_{}_{}.npy'.format(self.n_cls, dim, number))
        optimal_target_order = np.arange(self.n_cls)
        target_repeat = tr * np.ones(self.n_cls)

        optimal_target = torch.Tensor(optimal_target).float()
        target_repeat = torch.Tensor(target_repeat).long()
        optimal_target = torch.cat(
            [optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], dim=0)

        target_labels = torch.cat(
            [torch.Tensor([optimal_target_order[i]]).repeat(target_repeat[i]) for i in range(len(target_repeat))],
            dim=0).long().unsqueeze(-1)

        self.register_buffer("optimal_target", optimal_target)
        self.register_buffer("optimal_target_unique", optimal_target[::self.tr, :].contiguous().transpose(0, 1))
        self.register_buffer("target_labels", target_labels)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_labels", -torch.ones(1, K).long())
        self.register_buffer("class_centroid", torch.randn(self.n_cls, dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.class_centroid = nn.functional.normalize(self.class_centroid, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_labels[:, ptr:ptr + batch_size] = labels.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, feat_s_raw, feat_t_raw, im_labels, sep_pos_neg=False):
        """
        Input:
            feat_s_raw: a batch of features 1
            feat_t_raw: a batch of features 2
        Output:
            logits, targets
        """
        # compute logits
        # Einstein sum is more intuitive
        # positive logits from augmentation: Nx1
        q = F.normalize(feat_s_raw.float(), dim=-1)
        k = F.normalize(feat_t_raw.float(), dim=-1)
        im_labels = im_labels.cuda()
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if self.targeted:
            queue_negatives = self.queue.clone().detach()
            target_negatives = self.optimal_target.transpose(0, 1)
            l_neg = torch.einsum('nc,ck->nk', [q, torch.cat([queue_negatives, target_negatives], dim=1)])
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # positive logits from queue
        im_labels = im_labels.contiguous().view(-1, 1)

        if self.targeted:
            queue_labels = self.queue_labels.clone().detach()
            target_labels = self.target_labels.transpose(0, 1)

            # compute the optimal matching that minimize moving distance between memory bank anchors and targets
            with torch.no_grad():
                mask = torch.eq(im_labels, torch.cat([queue_labels, torch.full_like(target_labels, -1).to(queue_labels.device)], dim=1)).float()
                # im_labels_all = concat_all_gather(im_labels)
                # features_all = concat_all_gather(q.detach())
                
                im_labels_all = im_labels
                features_all = q.detach()
                
                # update memory bank class centroids
                for one_label in torch.unique(im_labels_all):
                    class_centroid_batch = F.normalize(torch.mean(features_all[im_labels_all[:, 0].eq(one_label), :], dim=0), dim=0)
                    self.class_centroid[one_label] = 0.9*self.class_centroid[one_label] + 0.1*class_centroid_batch
                    self.class_centroid[one_label] = F.normalize(self.class_centroid[one_label], dim=0)

                centroid_target_dist = torch.einsum('nc,ck->nk', [self.class_centroid, self.optimal_target_unique])
                centroid_target_dist = centroid_target_dist.detach().cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(-centroid_target_dist)

                for one_label, one_idx in zip(row_ind, col_ind):
                    if one_label not in im_labels:
                        continue
                    one_indices = torch.Tensor([i+one_idx*self.tr for i in range(self.tr)]).long()
                    tmp = mask[im_labels[:, 0].eq(one_label), :]
                    tmp[:, queue_labels.size(1)+one_indices] = 1
                    mask[im_labels[:, 0].eq(one_label), :] = tmp

            # separate samples and target
            if self.sep_t:
                mask_target = mask.clone()
                mask_target[:, :queue_labels.size(1)] = 0
                mask[:, queue_labels.size(1):] = 0
        else:
            mask = torch.eq(im_labels, self.queue_labels.clone().detach()).float()
        mask_pos_view = torch.zeros_like(mask)

        # sample num_positive from each class
        if self.num_positive > 0:
            for i in range(self.num_positive):
                all_pos_idxs = mask.view(-1).nonzero().view(-1)
                num_pos_per_anchor = mask.sum(1)
                num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
                num_pos_cum[0] = 0
                rand = torch.rand(mask.size(0), device=mask.device)
                idxs = ((rand * num_pos_per_anchor).floor() + num_pos_cum).long()
                idxs = idxs[num_pos_per_anchor.nonzero().view(-1)]
                sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
                mask_pos_view.view(-1)[sampled_pos_idxs] = 1
                mask.view(-1)[sampled_pos_idxs] = 0
        else:
            mask_pos_view = mask.clone()

        if self.targeted and self.sep_t:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_target.clone()
            mask_pos_view += mask_target
        else:
            mask_pos_view_class = mask_pos_view.clone()
            mask_pos_view_target = mask_pos_view.clone()
            mask_pos_view_class[:, self.queue_labels.size(1):] = 0
            mask_pos_view_target[:, :self.queue_labels.size(1)] = 0

        mask_pos_view = torch.cat([torch.ones([mask_pos_view.shape[0], 1]).cuda(), mask_pos_view], dim=1)
        mask_pos_view_class = torch.cat([torch.ones([mask_pos_view_class.shape[0], 1]).cuda(), mask_pos_view_class], dim=1)
        mask_pos_view_target = torch.cat([torch.zeros([mask_pos_view_target.shape[0], 1]).cuda(), mask_pos_view_target], dim=1)

        # apply temperature
        # logits /= self.T

        # log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
        log_prob = logits
        
        loss_class = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0] \
                    + torch.sum(((1 - mask_pos_view_class) * log_prob).sum(1) / (1-mask_pos_view).sum(1)) / (1-mask_pos_view).shape[0]
        loss_target = - torch.sum((mask_pos_view_target * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0] \
                    + torch.sum(((1 - mask_pos_view_target) * log_prob).sum(1) / (1-mask_pos_view).sum(1)) / (1-mask_pos_view).shape[0]
        loss_target = loss_target * self.tw
        loss_logit = F.cross_entropy(logits, labels)
        loss = loss_class + loss_target + loss_logit
        

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, im_labels)
        
        # if sep_pos_neg:
        #     pos_loss = (log_prob * mask_pos_view_class).mean(dim=-1)+(log_prob * mask_pos_view_target).mean(dim=-1)* self.tw
        #     neg_loss = (log_prob * (1- mask_pos_view_class)).mean(dim=-1)+(log_prob * (1- mask_pos_view_target)).mean(dim=-1)* self.tw
        #     return (pos_loss, neg_loss)
            
        return loss
    
    
    
# class TSC_half(nn.Module):
#     """
#     Build a MoCo model with: a query encoder, a key encoder, and a queue
#     https://arxiv.org/abs/1911.05722
#     """
#     def __init__(self, dim=1024, K=65536, m=0.999, T=0.07, mlp=False, num_positive=0, targeted=False, tr=1,
#                  sep_t=False, tw=1, number=0):
#         """
#         dim: feature dimension (default: 1024)
#         K: queue size; number of negative keys (default: 65536)
#         m: moco momentum of updating key encoder (default: 0.999)
#         T: softmax temperature (default: 0.07)
#         """
#         super(TSC_half, self).__init__()

#         self.K = K
#         self.m = m
#         self.T = T
#         self.num_positive = num_positive
#         self.n_cls = 2
#         self.targeted = targeted
#         self.tr = tr
#         self.sep_t = sep_t
#         self.tw = tw
#         optimal_target = np.load('optimal_{}_{}_{}.npy'.format(self.n_cls, dim, number))
#         optimal_target_order = np.arange(self.n_cls)
#         target_repeat = tr * np.ones(self.n_cls)

#         optimal_target = torch.Tensor(optimal_target).float()
#         target_repeat = torch.Tensor(target_repeat).long()
#         optimal_target = torch.cat(
#             [optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], dim=0)

#         target_labels = torch.cat(
#             [torch.Tensor([optimal_target_order[i]]).repeat(target_repeat[i]) for i in range(len(target_repeat))],
#             dim=0).long().unsqueeze(-1)

#         self.register_buffer("optimal_target", optimal_target)
#         self.register_buffer("optimal_target_unique", optimal_target[::self.tr, :].contiguous().transpose(0, 1))
#         self.register_buffer("target_labels", target_labels)

#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, K))
#         self.register_buffer("queue_labels", -torch.ones(1, K).long())
#         self.register_buffer("class_centroid", torch.randn(self.n_cls, dim))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#         self.class_centroid = nn.functional.normalize(self.class_centroid, dim=1)

#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys, labels):

#         batch_size = keys.shape[0]

#         ptr = int(self.queue_ptr)
#         assert self.K % batch_size == 0  # for simplicity

#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr:ptr + batch_size] = keys.T
#         self.queue_labels[:, ptr:ptr + batch_size] = labels.T
#         ptr = (ptr + batch_size) % self.K  # move pointer

#         self.queue_ptr[0] = ptr

#     @torch.no_grad()
#     def _batch_shuffle_ddp(self, x):
#         """
#         Batch shuffle, for making use of BatchNorm.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         # x_gather = concat_all_gather(x)
#         x_gather = x
#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # random shuffle index
#         idx_shuffle = torch.randperm(batch_size_all).cuda()

#         # broadcast to all gpus
#         torch.distributed.broadcast(idx_shuffle, src=0)

#         # index for restoring
#         idx_unshuffle = torch.argsort(idx_shuffle)

#         # shuffled index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

#         return x_gather[idx_this], idx_unshuffle

#     @torch.no_grad()
#     def _batch_unshuffle_ddp(self, x, idx_unshuffle):
#         """
#         Undo batch shuffle.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         # x_gather = concat_all_gather(x)
#         x_gather = x
#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # restored index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

#         return x_gather[idx_this]

#     def forward(self, feat_s_raw, feat_t_raw, im_labels, sep_pos_neg=False):
#         """
#         Input:
#             feat_s_raw: a batch of features 1
#             feat_t_raw: a batch of features 2
#         Output:
#             logits, targets
#         """
#         # compute logits
#         # Einstein sum is more intuitive
#         # positive logits from augmentation: Nx1
#         q = F.normalize(feat_s_raw.float(), dim=-1)
#         k = F.normalize(feat_t_raw.float(), dim=-1)
#         im_labels = im_labels.cuda()
        
#         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
#         # negative logits: NxK
#         if self.targeted:
#             queue_negatives = self.queue.clone().detach()
#             target_negatives = self.optimal_target.transpose(0, 1)
#             l_neg = torch.einsum('nc,ck->nk', [q, torch.cat([queue_negatives, target_negatives], dim=1)])
#         else:
#             l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

#         # logits: Nx(1+K)
#         logits = torch.cat([l_pos, l_neg], dim=1)

#         # labels: positive key indicators
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

#         # positive logits from queue
#         im_labels = im_labels.contiguous().view(-1, 1)

#         if self.targeted:
#             queue_labels = self.queue_labels.clone().detach()
#             target_labels = self.target_labels.transpose(0, 1)

#             # compute the optimal matching that minimize moving distance between memory bank anchors and targets
#             with torch.no_grad():
#                 mask = torch.eq(im_labels, torch.cat([queue_labels, torch.full_like(target_labels, -1).to(queue_labels.device)], dim=1)).float()
#                 # im_labels_all = concat_all_gather(im_labels)
#                 # features_all = concat_all_gather(q.detach())
                
#                 im_labels_all = im_labels
#                 features_all = q.detach()
                
#                 # update memory bank class centroids
#                 for one_label in torch.unique(im_labels_all):
#                     class_centroid_batch = F.normalize(torch.mean(features_all[im_labels_all[:, 0].eq(one_label), :], dim=0), dim=0)
#                     self.class_centroid[one_label] = 0.9*self.class_centroid[one_label] + 0.1*class_centroid_batch
#                     self.class_centroid[one_label] = F.normalize(self.class_centroid[one_label], dim=0)

#                 centroid_target_dist = torch.einsum('nc,ck->nk', [self.class_centroid, self.optimal_target_unique])
#                 centroid_target_dist = centroid_target_dist.detach().cpu().numpy()

#                 row_ind, col_ind = linear_sum_assignment(-centroid_target_dist)

#                 for one_label, one_idx in zip(row_ind, col_ind):
#                     if one_label not in im_labels:
#                         continue
#                     one_indices = torch.Tensor([i+one_idx*self.tr for i in range(self.tr)]).long()
#                     tmp = mask[im_labels[:, 0].eq(one_label), :]
#                     tmp[:, queue_labels.size(1)+one_indices] = 1
#                     mask[im_labels[:, 0].eq(one_label), :] = tmp

#             # separate samples and target
#             if self.sep_t:
#                 mask_target = mask.clone()
#                 mask_target[:, :queue_labels.size(1)] = 0
#                 mask[:, queue_labels.size(1):] = 0
#         else:
#             mask = torch.eq(im_labels, self.queue_labels.clone().detach()).float()
#         mask_pos_view = torch.zeros_like(mask)

#         # sample num_positive from each class
#         if self.num_positive > 0:
#             for i in range(self.num_positive):
#                 all_pos_idxs = mask.view(-1).nonzero().view(-1)
#                 num_pos_per_anchor = mask.sum(1)
#                 num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
#                 num_pos_cum[0] = 0
#                 rand = torch.rand(mask.size(0), device=mask.device)
#                 idxs = ((rand * num_pos_per_anchor).floor() + num_pos_cum).long()
#                 idxs = idxs[num_pos_per_anchor.nonzero().view(-1)]
#                 sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
#                 mask_pos_view.view(-1)[sampled_pos_idxs] = 1
#                 mask.view(-1)[sampled_pos_idxs] = 0
#         else:
#             mask_pos_view = mask.clone()

#         if self.targeted and self.sep_t:
#             mask_pos_view_class = mask_pos_view.clone()
#             mask_pos_view_target = mask_target.clone()
#             mask_pos_view += mask_target
#         else:
#             mask_pos_view_class = mask_pos_view.clone()
#             mask_pos_view_target = mask_pos_view.clone()
#             mask_pos_view_class[:, self.queue_labels.size(1):] = 0
#             mask_pos_view_target[:, :self.queue_labels.size(1)] = 0

#         mask_pos_view = torch.cat([torch.ones([mask_pos_view.shape[0], 1]).cuda(), mask_pos_view], dim=1)
#         mask_pos_view_class = torch.cat([torch.ones([mask_pos_view_class.shape[0], 1]).cuda(), mask_pos_view_class], dim=1)
#         mask_pos_view_target = torch.cat([torch.zeros([mask_pos_view_target.shape[0], 1]).cuda(), mask_pos_view_target], dim=1)

#         # apply temperature
#         # logits /= self.T

#         # log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
#         log_prob = logits
        
#         loss_class = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0] \
#                     + torch.sum(((1 - mask_pos_view_class) * log_prob).sum(1) / (1-mask_pos_view).sum(1)) / (1-mask_pos_view).shape[0]
#         loss_target = - torch.sum((mask_pos_view_target * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0] \
#                     + torch.sum(((1 - mask_pos_view_target) * log_prob).sum(1) / (1-mask_pos_view).sum(1)) / (1-mask_pos_view).shape[0]
#         loss_target = loss_target * self.tw
#         loss_logit = F.cross_entropy(logits, labels)
#         loss = loss_class + loss_target + loss_logit
        

#         # dequeue and enqueue
#         self._dequeue_and_enqueue(k, im_labels)
        
#         # if sep_pos_neg:
#         #     pos_loss = (log_prob * mask_pos_view_class).mean(dim=-1)+(log_prob * mask_pos_view_target).mean(dim=-1)* self.tw
#         #     neg_loss = (log_prob * (1- mask_pos_view_class)).mean(dim=-1)+(log_prob * (1- mask_pos_view_target)).mean(dim=-1)* self.tw
#         #     return (pos_loss, neg_loss)
            
#         return loss