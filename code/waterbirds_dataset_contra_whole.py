import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
import torch.nn as nn
from models.initializer import initialize_model
from configs.utils import ParseKwargs, parse_bool, populate_defaults
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
import copy

def my_filtering_function(pair):
    wanted_key = 'model'
    key, value = pair
    if wanted_key in key :
        return True  # filter pair out of the dictionary
    else:
        return False
        
class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


class WaterbirdsDataset(WILDSDataset):
    """
    The Waterbirds dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to facilitate comparisons to previous work.

    Supported `split_scheme`:
        'official'

    Input (x):
        Images of birds against various backgrounds that have already been cropped and centered.

    Label (y):
        y is binary. It is 1 if the bird is a waterbird (e.g., duck), and 0 if it is a landbird.

    Metadata:
        Each image is annotated with whether the background is a land or water background.

    Original publication:
        @inproceedings{sagawa2019distributionally,
          title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle = {International Conference on Learning Representations},
          year = {2019}
        }

    The dataset was constructed from the CUB-200-2011 dataset and the Places dataset:
        @techreport{WahCUB_200_2011,
        	Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
        	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
        	Year = {2011}
        	Institution = {California Institute of Technology},
        	Number = {CNS-TR-2011-001}
        }
        @article{zhou2017places,
          title = {Places: A 10 million Image Database for Scene Recognition},
          author = {Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
          journal ={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          year = {2017},
          publisher = {IEEE}
        }

    License:
        The use of this dataset is restricted to non-commercial research and educational purposes.
    """

    _dataset_name = 'waterbirds'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', 
                 model=None, positive_k=16, negative_k=16):
        self._version = version
        self.positive_k = positive_k if isinstance(positive_k, int) else int(positive_k)
        self.negative_k = negative_k if isinstance(negative_k, int) else int(negative_k)
        self._data_dir = self.initialize_data_dir(root_dir, download)

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        # Note: metadata_df is one-indexed.
        metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self._y_array = torch.LongTensor(metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df['place'].values), self._y_array),
            dim=1
        )
        self._metadata_fields = ['generic-spurious', 'y']
        self._metadata_map = {
            'generic-spurious': ['bird on land', 'bird on water'], # Padding for str formatting
            'spurious': ['land', 'water'], 
            'y': ['landbird', 'waterbird']
        }

        # Extract filenames
        self._input_array = metadata_df['img_filename'].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = metadata_df['split'].values

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['generic-spurious', 'y']))

        super().__init__(root_dir, download, split_scheme)
        weights = [[1.0] for _ in range(len(self))]
        self.weights = torch.from_numpy(np.array(weights))
        # print('self.weights', self.weights.shape) # self.weights torch.Size([11788, 1])
        # exit()
        self.model = model
        

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       img_filename = os.path.join(
           self.data_dir,
           self._input_array[idx])
       x = Image.open(img_filename).convert('RGB')
       return x

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)

        results, results_str = self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

        # For Waterbirds, the validation and test sets are constructed to be more balanced
        # compared to the training set.
        # To compute the actual average accuracy over the empirical (training) distribution,
        # we therefore weight each groups according to their frequency in the training set.

        results['adj_acc_avg'] = (
            (results['acc_y:landbird_generic-spurious:birdonland'] * 3498
            + results['acc_y:landbird_generic-spurious:birdonwater'] * 184
            + results['acc_y:waterbird_generic-spurious:birdonland'] * 56
            + results['acc_y:waterbird_generic-spurious:birdonwater'] * 1057) /
            (3498 + 184 + 56 + 1057))

        del results['acc_avg']
        results_str = f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n" + '\n'.join(results_str.split('\n')[1:])

        return results, results_str
    
    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self._metadata_array[idx]
        return x, y, metadata, idx, self.weights[idx]
    
    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
        print('self.model', self.model)
        return WILDSSubset_self(self, split_idx, transform, model=self.model, pos=self.positive_k, neg=self.negative_k)
    

class TruncateDataset(WILDSDataset):
    _dataset_name = 'waterbirds'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official', transform=None):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self.transform = transform
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        # Note: metadata_df is one-indexed.
        metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'one_metadata.csv'))

        # Get the y values
        self._y_array = torch.LongTensor(metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df['place'].values), self._y_array),
            dim=1
        )
        self._metadata_fields = ['generic-spurious', 'y']
        self._metadata_map = {
            'generic-spurious': ['bird on land', 'bird on water'], # Padding for str formatting
            'spurious': ['land', 'water'], 
            'y': ['landbird', 'waterbird']
        }

        # Extract filenames
        self._input_array = metadata_df['img_filename'].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = metadata_df['split'].values

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['generic-spurious', 'y']))

        super().__init__(root_dir, download, split_scheme)
        weights = [[1.0] for _ in range(len(self))]
        self.weights = torch.from_numpy(np.array(weights))

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       img_filename = os.path.join(
           self.data_dir,
           self._input_array[idx])
       x = Image.open(img_filename).convert('RGB')
       return x
   
    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        idx = torch.remainder(torch.tensor(idx), torch.tensor(len(self)))
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self._metadata_array[idx]
        return x, y, metadata, idx, self.weights[idx]
    

class WILDSSubset_self(WILDSDataset):
    def __init__(self, dataset, indices, transform, do_transform_y=False, model=None, pos=16, neg=16):
        """
        This acts like `torch.utils.data.Subset`, but on `WILDSDatasets`.
        We pass in `transform` (which is used for data augmentation) explicitly
        because it can potentially vary on the training vs. test subsets.

        `do_transform_y` (bool): When this is false (the default),
                                 `self.transform ` acts only on  `x`.
                                 Set this to true if `self.transform` should
                                 operate on `(x,y)` instead of just `x`.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_scheme', '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y
        self.model = model
        self.pos=pos
        self.neg=neg
        print('self.model', self.model)
        d = edict({
            'model':'clip-rn50', 
            'load_featurizer_only':False, 
            'dataset':'waterbirds', 
            'algorithm': 'Multimodal',
            'groupby_fields': 'y', 
            'model_kwargs': {},
            'loader_kwargs': {},
            'dataset_kwargs': {}, 
            'version': '1.0',
            'root_dir': '/data/data/CLIP-spurious-finetune/data',
            'download': True, 
            'split_scheme': 'official', 
            'imagenet_class':'baby pacifier',
            'seed': 0,
            'dropout_before': 0.0,
            'freeze_vision': True,
            'freeze_language': True, 
            'train_projection': False, 
            'num_templates': 'all', 
            'finetuning': 'zeroshot'
        })
        self.contra_dataset = TruncateDataset(version='1.0', root_dir='/data/data/CLIP-spurious-finetune/data', transform=transform)
        self.wrong_idxes = torch.load('/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds/all_wrong_predict_idxes.pkl')
        if self.model is None or 'vit' not in self.model:
            erm_model = initialize_model(d, d_out=None).cuda()
            
            original = torch.load('/data/data/CLIP-spurious-finetune/logs/clip-rn50_waterbirds_TSC/ERM_freeze-language_freeze-vision_train-projection_lr_1e-04_wd_1e-05_batchsize_128_seed_11111111/waterbirds_seed:11111111_epoch:best_model.pth')\
                    ['algorithm']
            state_dict = dict(filter(my_filtering_function, original.items()))
            modified = {}
            for k, v in state_dict.items():
                modified[k[6:]] = v
            erm_model.load_state_dict(modified)
            erm_model.eval()
            self.erm_model = erm_model.cuda()
            correct_predict_embeddings = torch.load('/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds/correct_predict_embeddings.pkl')
            correct_predict_embeddings[0] = torch.cat(correct_predict_embeddings[0])
            correct_predict_embeddings[1] = torch.cat(correct_predict_embeddings[1])
            self.correct_predict_embeddings = correct_predict_embeddings
            all_class_embeddings = torch.load('/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds/all_class_embeddings.pkl')
            all_class_embeddings[0] = torch.cat(all_class_embeddings[0])
            all_class_embeddings[1] = torch.cat(all_class_embeddings[1])
            self.all_class_embeddings = all_class_embeddings
        else:
            d.model = 'clip-vit'
            erm_model = initialize_model(d, d_out=None).cuda()
            erm_model.eval()
            self.erm_model = erm_model.cuda()
            self.erm_model = erm_model.cuda()
            correct_predict_embeddings = torch.load('/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds_vit/correct_predict_embeddings.pkl')
            correct_predict_embeddings[0] = torch.cat(correct_predict_embeddings[0])
            correct_predict_embeddings[1] = torch.cat(correct_predict_embeddings[1])
            self.correct_predict_embeddings = correct_predict_embeddings
            all_class_embeddings = torch.load('/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds_vit/all_class_embeddings.pkl')
            all_class_embeddings[0] = torch.cat(all_class_embeddings[0])
            all_class_embeddings[1] = torch.cat(all_class_embeddings[1])
            self.all_class_embeddings = all_class_embeddings
        
        # self.k = 16
        negative_embeddings = []
        for idx in tqdm(range(len(self.contra_dataset))):
            x, y, metadata, idxes, weights = self.contra_dataset[idx]
            if self.transform is not None:
                if self.do_transform_y:
                    x, y = self.transform(x, y)
                else:
                    x = self.transform(x)
            x = x.cuda()
            x = x.unsqueeze(0)
            
            if self.model is None or 'vit' not in self.model:
                layer = getattr(self.erm_model.model.visual, 'layer4')
                with Hook(layer) as hook:
                    _, _ = self.erm_model(x, True)
                    embedding = hook.activation.float().reshape(1, -1)
            else:
                _, _ = erm_model(x, True)
                embedding = erm_model.model.visual.observer
            counter_label = torch.abs(1-y)
            hard_embeddings = self.all_class_embeddings[counter_label.item()]
            cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(embedding, hard_embeddings.cuda())
            _, topk = torch.topk(cos_sim, k=self.neg, dim=-1)
            negative = hard_embeddings[topk]
            negative_embeddings.append(negative.unsqueeze(0))
        self.negative_embeddings = torch.cat(negative_embeddings)

    def __getitem__(self, idx):
        idx_partial = torch.remainder(torch.tensor(idx), len(self.negative_embeddings))
        contra_x, contra_y, _, idxes_partial, weights_partial = self.contra_dataset[idx_partial][:5]
        x, y, metadata, idxes, weights = self.dataset[self.indices[idx]]
        
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
                contra_x, contra_y = self.transform(contra_x, contra_y)
            else:
                x = self.transform(x)
                contra_x = self.transform(contra_x)
                
        else:
            print('error')
            x = x.resize((self.erm_model.model.visual.input_resolution, 
                          self.erm_model.model.visual.input_resolution))
            x = torch.tensor(np.array(x).astype(np.float32) / 255.)
            x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2).float().cuda()
        # print('x shape', x.shape)
        x = x.cuda()
        
        correct_embeddings = self.correct_predict_embeddings[contra_y.item()]
        perm = torch.randperm(correct_embeddings.size(0))
        chosen_correct = perm[:self.pos] # positive
        positive = correct_embeddings[chosen_correct]
        
        idx = torch.remainder(torch.tensor(idx_partial), torch.tensor(len(self.negative_embeddings)))
        negative = self.negative_embeddings[idx]
        
        negative_embeddings = self.all_class_embeddings[contra_y.item()]
        perm = torch.randperm(negative_embeddings.size(0))
        chosen_correct = perm[:self.neg] # positive
        negative = negative_embeddings[chosen_correct]
        
        return x, y, metadata, idxes, weights, contra_x, contra_y, positive, negative, weights_partial, idxes_partial

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)


if __name__ == '__main__':
    dataset = WaterbirdsDataset('1.0', root_dir='/data/data/CLIP-spurious-finetune/data')
    train_data = dataset.get_subset('train')
    for idx in range(len(train_data)):
        x, y, metadata, idxes, weights, contra_x, positive, negative, wrong = train_data[idx]
        print('positive', positive.shape)