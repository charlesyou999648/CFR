from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to
import torch

class ERM(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out)
        self.erm_coef = config.erm_coef
        self.tau = config.weights
        for param in model.parameters():
            # Check if parameter dtype is  Half (float16)
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
        x, y_true, metadata, idxes, weights, contra_x, positive, negative, wrong = batch
        # add contra_x: fully: x and contra_x different; partial: x and contra_x different/same
        bs_ = x.shape[0]
        # print('')
        contra_x = move_to(contra_x, self.device)
        # if self.training:
        #     assert (x == contra_x).all() 
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        # outputs = self.get_model_output(x, y_true)
        # outputs, features = self.model(x, return_features=True)
        outputs, _ = self.model(x, return_features=True)
        
        # print('y_pred', outputs.shape)
        # print('features', features.shape)
        # print('positive', positive.shape)
        # print('negative', negative.shape)
        
        _, features = self.model(contra_x, return_features=True)
        
        # print('y_pred', outputs.shape)
        # print('features', features.shape)
        # print('positive', positive.shape)
        # print('negative', negative.shape)
        
        positive = positive.flatten(0, 1).cuda().float()
        negative = negative.flatten(0, 1).cuda().float()
        positive = self.model.model.visual.attnpool.forward(positive.reshape(positive.shape[0], 2048, 7, 7))
        negative = self.model.model.visual.attnpool.forward(negative.reshape(negative.shape[0], 2048, 7, 7))
        
        positive = positive.reshape(bs_, -1, 1024)
        negative = negative.reshape(bs_, -1, 1024)
        # print('positive', positive.shape)
        # print('negative', negative.shape)
        features = torch.nn.functional.normalize(features, dim=-1)
        positive = torch.nn.functional.normalize(positive, dim=-1)
        negative = torch.nn.functional.normalize(negative, dim=-1)
        # exit()

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'positive': positive,
            'negative': negative,
            'anchor': features,
        }
        return results
    
    def contra_loss(self, results):
        positive = results['positive']
        negative = results['negative']
        anchor = results['anchor']
        loss=0.0
        for idx in range(positive.shape[1]):
            pos = positive[:, idx:idx+1]
            all_feat = torch.cat([pos, negative], dim=1)
            sim = torch.einsum('bd,bjd->bj', anchor, all_feat)/self.tau
            lbl = torch.zeros(pos.shape[0]).long().to(sim.device)
            loss += torch.nn.CrossEntropyLoss(reduction='mean')(sim, lbl)
        return loss/positive.shape[1]
    
    # def contra_loss(self, results):
    #     positive = results['positive']
    #     to_extend = positive.shape[1]
    #     negative = results['negative']
    #     anchor = results['anchor']
    #     tau = self.tau

    #     a = anchor.unsqueeze(1).unsqueeze(1)
    #     a = a.repeat(1, to_extend, 1, 1)
    #     pos = positive.unsqueeze(2)
    #     neg = negative.unsqueeze(1).repeat(1, to_extend, 1, 1)
    #     matrix = torch.cat([a, pos, neg], dim=2)
    #     matrix = matrix.flatten(0, 1)
    #     # print(matrix.shape)
    #     a = a.flatten(0, 1)
    #     # print("shape(anchor):", a.shape)
    #     logits = torch.einsum('bik,bjk->bij', a, matrix)
    #     logits = (torch.div(logits, tau))
    #     logits_max, _ = torch.max(logits, dim=2, keepdim=True)
    #     # print("shape(logit max):", logits_max.shape)
    #     logits = logits-logits_max.detach()
    #     # print("shape(logits):", logits.shape)
    #     mask1 = torch.zeros_like(logits)
    #     mask1[:, 0, 1] = 1
    #     mask2 = torch.ones_like(logits)
    #     mask2[:, 0, 0] = 0
    #     numerator = (mask1*logits).sum(2)
    #     denominator = (torch.exp(mask2*logits)).sum(2)
    #     # print("numerator:", numerator.shape)
    #     # print("shape(denominator):", denominator.shape)
    #     # print("denominator:", denominator[0])
    #     res = (-(numerator) + (torch.log(denominator+1e-6))).mean()
    #     # print(res)
    #     return res

    def objective(self, results):
        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        contra_loss = self.contra_loss(results)
        
        self.save_metric_for_logging(
            results, "classification_loss", labeled_loss
        )
        self.save_metric_for_logging(
            results, "contrastive_loss", contra_loss
        )
        
        return self.erm_coef*labeled_loss + contra_loss