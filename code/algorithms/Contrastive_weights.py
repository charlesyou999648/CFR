from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to
import torch
from TSC import TSC
import numpy as np
from clip import ModifiedResNet, VisionTransformer

class CFR(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out)
        self.config = config
        self.erm_coef = 1.0
        self.tau = 1 
        for param in model.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)
        # initialize module
        super().__init__(
            config=config,
            model=model.cuda(),
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        if 'vit' in self.config.model:
            dim= 768
        else:
            dim=1024
        self.tsc = TSC(dim=dim, K=128*2*5, targeted=True, tw=0.2, number=0, n_cls=2).cuda()
        
    def similarity_loss(self, features_1, features_2, labels_1, labels_2, sep_pos_neg=False, abs=True):
        loss = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(features_1.unsqueeze(1), features_2.unsqueeze(0))

        pos_mask = (labels_1.unsqueeze(1).expand(-1, len(labels_2))-\
            labels_2.unsqueeze(0).expand(len(labels_1),-1)) == 0
        pos_mask = pos_mask.float().cuda()

        pos_loss = (loss * pos_mask).mean(dim=-1)
        neg_loss = (loss * (1- pos_mask)).mean(dim=-1)
        if abs:
            neg_loss = neg_loss.abs()

        if sep_pos_neg:
            return (pos_loss, neg_loss)
        else:
            return -  pos_loss +  neg_loss

    def process_batch(self, batch, epoch=0):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch 
        """
        x, y_true, metadata, idxes, weights, contra_x, contra_y, positive, negative, weights_partial, idxes_partial = batch
        bs_ = x.shape[0]
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        contra_y = move_to(contra_y, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        weights_partial = move_to(weights_partial, self.device)

        outputs_full, feat_full  = self.model(x, return_features=True, dropout=self.config.dropout)
                
        outputs, features = self.model(contra_x.cuda(), return_features=True, dropout=self.config.dropout)
        
        positive = move_to(positive, self.device).float()
        negative = move_to(negative, self.device).float()
        if isinstance(self.model.model.visual, ModifiedResNet):
            positive = self.model.model.visual.attnpool.forward(positive.reshape(-1, 2048, 7, 7))
            negative = self.model.model.visual.attnpool.forward(negative.reshape(-1, 2048, 7, 7))
            
            positive = positive.reshape(bs_, -1, 1024)
            negative = negative.reshape(bs_, -1, 1024)
        else:
            positive = positive.reshape(-1, 1024) @ self.model.model.visual.proj
            negative = negative.reshape(-1, 1024) @ self.model.model.visual.proj
            
            positive = positive.reshape(bs_, -1, 768)
            negative = negative.reshape(bs_, -1, 768)
        features = torch.nn.functional.normalize(features, dim=-1)
        positive = torch.nn.functional.normalize(positive, dim=-1)
        negative = torch.nn.functional.normalize(negative, dim=-1)
        
        # exit()
        # if epoch >= self.config.n_epochs//10:
        #     if not self.config.one_divide:
        #         weights = 1+10*weights
        #     else:
        #         weights = 0.25/(weights+1e-10)
        #     weights[weights>self.config.weight_high] = self.config.weight_high
        #     weights[weights<self.config.weight_low] = self.config.weight_low
            
        #     if self.config.one_divide:
        #         weights = weights+1
                
            # if not self.config.partial_one_divide:
            #     weights_partial = 1+10*weights_partial
            # else:
            #     weights_partial = 0.25/(weights_partial+1e-10)
            # weights_partial[weights_partial>self.config.partial_weight_high] = self.config.partial_weight_high
            # weights_partial[weights_partial<self.config.partial_weight_high] = self.config.partial_weight_high
            
            # if self.config.partial_one_divide:
            #     weights_partial = weights_partial+1

        results = {
            'g': g,
            'y_true_partial': contra_y,
            'y_true': y_true,
            'y_pred_partial': outputs,
            'y_pred': outputs_full,
            'feat_full': feat_full,
            'metadata': metadata,
            'positive': positive,
            'negative': negative,
            'anchor': features,
            'weights': weights,
            'weights_partial': weights_partial,
            'epoch': epoch,
        }
        return results
    
    def contra_loss(self, results):
        positive = results['positive']
        negative = results['negative']
        anchor = results['anchor']
        weights_partial = results['weights_partial']
        loss=0.0
        for idx in range(positive.shape[1]):
            pos = positive[:, idx:idx+1]
            all_feat = torch.cat([pos, negative], dim=1)
            sim = torch.einsum('bd,bjd->bj', anchor, all_feat)/self.tau
            lbl = torch.zeros(pos.shape[0]).long().to(sim.device)
            loss += (torch.nn.CrossEntropyLoss(reduction='none')(sim, lbl)*weights_partial).mean()
        return loss/positive.shape[1]
    

    def objective(self, results):
        labeled_loss = self.loss.compute(results['y_pred_partial'], results['y_true_partial'], return_dict=False)
        contra_loss = self.contra_loss(results)
        image_loss = self.similarity_loss(results['y_pred'], results['y_pred'], results['y_true'], results['y_true'], abs=False)
        if self.training:
            image_loss = self.tsc(results['feat_full'], results['feat_full'], results['y_true'], 0.0)
        image_loss = (image_loss*results['weights'].cuda()).mean()
        self.save_metric_for_logging(
            results, "classification_loss", labeled_loss
        )
        self.save_metric_for_logging(
            results, "contrastive_loss", contra_loss
        )
        self.save_metric_for_logging(
            results, "image_loss", image_loss
        )
        
        ep = results['epoch']
        coef = 1  + 0.5*torch.cos(torch.tensor(2*np.pi*ep/self.config.n_epochs))
        lab_coef = 1 + 0.5*torch.cos(torch.tensor(2*np.pi*ep/self.config.n_epochs))
        
        return lab_coef*labeled_loss + coef*contra_loss + image_loss