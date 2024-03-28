# # # # %%
import urllib.request
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
from PIL import Image
from scipy.ndimage import filters
from torch import nn
from tqdm import tqdm

# # # %%
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


# # # %%
from models.initializer import initialize_model, get_dataset
from configs.utils import ParseKwargs, parse_bool, populate_defaults
# from run_expt_original import config
import argparse
parser = argparse.ArgumentParser('')
parser.add_argument('--model', default='clip-rn50')
parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')
parser.add_argument('-d', '--dataset', default='waterbirds')
parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')
parser.add_argument('--root_dir', default='data')
parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                    help='If true, tries to download the dataset if it does not exist in root_dir.')
parser.add_argument('--algorithm', default='Multimodal')
parser.add_argument('--groupby_fields', default='y')
parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for model initialization passed as key1=value1 key2=value2')
parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')
parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
parser.add_argument('--n_groups_per_batch', type=int)
parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
parser.add_argument('--imagenet_class', type=str, default='baby pacifier', help='If use ImageNet-Spurious dataset, which ImageNet class to use.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--freeze_vision', type=parse_bool, const=True, default=False, nargs='?') 
parser.add_argument('--num_templates', type=str, default='all')
parser.add_argument('--freeze_language', type=parse_bool, const=True, default=True, nargs='?') 
parser.add_argument('--train_projection', type=parse_bool, const=True, default=True, nargs='?') 
parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                    help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
parser.add_argument('--finetuning', choices=['zeroshot', 'linear'], default='zeroshot')
parser.add_argument('--dropout_before', type=float, default=0.0)

config = parser.parse_args('')
config = populate_defaults(config)

# # %%
config.dataset = 'waterbirds'
# config.root_dir = '/data'
erm_model = initialize_model(config, d_out=None).cuda()

# # %%
def my_filtering_function(pair):
        wanted_key = 'model'
        key, value = pair
        if wanted_key in key :
            return True  # filter pair out of the dictionary
        else:
            return False
if 'waterbirds' in config.dataset:
        original = torch.load('/data/data/CLIP-spurious-finetune/logs/clip-rn50_waterbirds_TSC/ERM_freeze-language_freeze-vision_train-projection_lr_1e-04_wd_1e-05_batchsize_128_seed_11111111/waterbirds_seed:11111111_epoch:best_model.pth')\
                ['algorithm']
        state_dict = dict(filter(my_filtering_function, original.items()))
        modified = {}
        for k, v in state_dict.items():
            modified[k[6:]] = v
        erm_model.load_state_dict(modified)

# # # # %%
dataset = get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        imagenet_class=config.imagenet_class,
        seed=config.seed,
        **config.dataset_kwargs)

test_dataset = dataset.get_subset(split='train')
sample_idx=torch.load( '/data/data/CLIP-spurious-finetune/contra_adapt_cmnist/sample_idx.pkl')
# %%
print(len(test_dataset))

# # # # %%erm_model.eval()
all_embeddings = []
all_wrong_predict_idxes = []
correct_predict_embeddings = {k:[] for k in range(2)}
class_embeddings = {k:[] for k in range(2)}
layer = getattr(erm_model.model.visual, 'layer4')
# erm_model.model.visual.observer = nn.Linear(24, 1024)
# erm_model.model.visual.observer.weight = layer
# layer = getattr(erm_model.model.visual, 'observer')
# erm_model.model.visual.proj = None
with torch.no_grad():
    for idx in tqdm(range(len(test_dataset))):
        x, y_true, metadata, idxes, weights = test_dataset[idx][:5]
        x = x.resize((erm_model.model.visual.input_resolution, erm_model.model.visual.input_resolution))
        x = torch.tensor(np.array(x).astype(np.float32) / 255.).cuda()
        # print(x.shape)
        inp = x.unsqueeze(0)
        inp = inp.permute(0, 3, 1, 2).float()
        with Hook(layer) as hook:
            res, feat = erm_model(inp, True)
            # embedding = erm_model.model.visual.observer
            embedding = hook.activation.float().reshape(1, -1)
        pred = res.argmax(-1)
        if (pred - y_true).sum() != 0:
            all_wrong_predict_idxes.append(idxes)
        else:
            try:
                correct_predict_embeddings[y_true.item()].append(embedding.cpu())
            except:
                print(len(correct_predict_embeddings))
                print(y_true)
                break
        try:
            class_embeddings[y_true.item()].append(embedding.cpu())
        except:
            print(len(class_embeddings))
            print(y_true)
            break
        break

config.dataset = 'waterbirds'
erm_model = initialize_model(config, d_out=None).cuda()
dataset = get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        imagenet_class=config.imagenet_class,
        seed=config.seed,
        **config.dataset_kwargs)
# config.dataset = 'cmnist'
test_dataset = dataset.get_subset(split='train', transform=clip.load('RN50')[-1])

def my_filtering_function(pair):
        wanted_key = 'model'
        key, value = pair
        if wanted_key in key :
            return True  # filter pair out of the dictionary
        else:
            return False
if 'waterbirds' in config.dataset:
        original = torch.load('/data/data/CLIP-spurious-finetune/logs/clip-rn50_waterbirds_TSC/ERM_freeze-language_freeze-vision_train-projection_lr_1e-04_wd_1e-05_batchsize_128_seed_11111111/waterbirds_seed:11111111_epoch:best_model.pth')\
                ['algorithm']
        state_dict = dict(filter(my_filtering_function, original.items()))
        modified = {}
        for k, v in state_dict.items():
            modified[k[6:]] = v
        erm_model.load_state_dict(modified)
        
erm_model.eval()
all_embeddings = []
all_wrong_predict_idxes = []
correct_predict_embeddings = {k:[] for k in range(2)}
class_embeddings = {k:[] for k in range(2)}
layer = getattr(erm_model.model.visual, 'layer4')

# erm_model.model.visual.observer = nn.Linear(24, 1024)
# erm_model.model.visual.observer.weight = layer
# layer = getattr(erm_model.model.visual, 'observer')
# erm_model.model.visual.proj = None

group_count = {k:0 for k in range(4)}
group_correct = {k:0 for k in range(4)}

class_idxes = {k:[] for k in range(2)}

# sample_idx=torch.load( '/data/data/CLIP-spurious-finetune/contra_adapt_cmnist/sample_idx.pkl')
with torch.no_grad():
    for idx in tqdm(range(len(test_dataset))):
        x, y_true, metadata, idxes, weights = test_dataset[idx][:5]
        # x = x.resize((erm_model.model.visual.input_resolution, erm_model.model.visual.input_resolution))
        
        x = torch.tensor(np.array(x).astype(np.float32) / 255.).cuda()
        g = dataset._eval_grouper.metadata_to_group(metadata.unsqueeze(0))
        group_count[g.item()] += 1
        inp = x.unsqueeze(0).float()
        with Hook(layer) as hook:
            res, feat = erm_model(inp, True)
            embedding = hook.activation.float().reshape(1, -1)
        pred = res.argmax(-1)
        if (pred - y_true).sum() != 0:
            all_wrong_predict_idxes.append(idxes)
        else:
            group_correct[g.item()] += 1
            class_idxes[y_true.item()].append(idxes)
            # try:
            correct_predict_embeddings[y_true.item()].append(embedding.cpu())
        try:
            class_embeddings[y_true.item()].append(embedding.cpu())
        except:
            print(len(class_embeddings))
            print(y_true)
# for i in range(25):
#     print('group: \t', i, '\t', group_correct[i], '\t', group_count[i], '\t', group_correct[i]/group_count[i])
# # cmnist ERM, test set
# import random
# random.seed(0)
# for i in range(0, 5):
#     all_wrong_predict_idxes.extend(random.sample(class_idxes[i], 20))

# # # # # %%
torch.save(
    all_wrong_predict_idxes, '/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds/all_wrong_predict_idxes.pkl'
)

# # # # # %%
torch.save(
    correct_predict_embeddings, '/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds/correct_predict_embeddings.pkl'
)

# # # # # %%
torch.save(
    class_embeddings, '/data/data/CLIP-spurious-finetune/contra_adapt_waterbirds/all_class_embeddings.pkl'
)
