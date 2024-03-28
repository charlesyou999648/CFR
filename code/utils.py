import sys
import os
import csv
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import re

from torch.utils.data import DataLoader

try:
    import wandb
except ImportError as e:
    pass


def cross_entropy_with_logits_loss(input, soft_target):
    """
    Implementation of CrossEntropy loss using a soft target. Extension of BCEWithLogitsLoss to MCE.
    Normally, cross entropy loss is
        \sum_j 1{j == y} -log \frac{e^{s_j}}{\sum_k e^{s_k}} = -log \frac{e^{s_y}}{\sum_k e^{s_k}}
    Here we use
        \sum_j P_j *-log \frac{e^{s_j}}{\sum_k e^{s_k}}
    where 0 <= P_j <= 1
    Does not support fancy nn.CrossEntropy options (e.g. weight, size_average, ignore_index, reductions, etc.)
    Args:
    - input (N, k): logits
    - soft_target (N, k): targets for softmax(input); likely want to use class probabilities
    Returns:
    - losses (N, 1)
    """
    return torch.sum(- soft_target * torch.nn.functional.log_softmax(input, 1), 1)

def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom==0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom==0:
            return 0.
    else:
        raise ValueError('Type of curr_counts not recognized')
    prev_weight = prev_counts/denom
    curr_weight = curr_counts/denom
    return prev_weight*prev_avg + curr_weight*curr_avg

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/

def save_model(algorithm, epoch, best_val_metric, path):
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)

def load(module, path, device=None, tries=2):
    """
    Handles loading weights saved from this repo/model into an algorithm/model.
    Attempts to handle key mismatches between this module's state_dict and the loaded state_dict.
    Args:
        - module (torch module): module to load parameters for
        - path (str): path to .pth file
        - device: device to load tensors on
        - tries: number of times to run the match_keys() function
    """
    if device is not None:
        state = torch.load(path, map_location=device)
    else:
        state = torch.load(path)

    # Loading from a saved WILDS Algorithm object
    if 'algorithm' in state:
        prev_epoch = state['epoch']
        best_val_metric = state['best_val_metric']
        state = state['algorithm']
    # Loading from a pretrained SwAV model
    elif 'state_dict' in state:
        state = state['state_dict']
        prev_epoch, best_val_metric = None, None
    else:
        prev_epoch, best_val_metric = None, None

    # If keys match perfectly, load_state_dict() will work
    try: module.load_state_dict(state)
    except:
        # Otherwise, attempt to reconcile mismatched keys and load with strict=False
        module_keys = module.state_dict().keys()
        for _ in range(tries):
            state = match_keys(state, list(module_keys))
            module.load_state_dict(state, strict=False)
            leftover_state = {k:v for k,v in state.items() if k in list(state.keys()-module_keys)}
            leftover_module_keys = module_keys - state.keys()
            if len(leftover_state) == 0 or len(leftover_module_keys) == 0: break
            state, module_keys = leftover_state, leftover_module_keys
        if len(module_keys-state.keys()) > 0: print(f"Some module parameters could not be found in the loaded state: {module_keys-state.keys()}")
    return prev_epoch, best_val_metric

def match_keys(d, ref):
    """
    Matches the format of keys between d (a dict) and ref (a list of keys).
    Helper function for situations where two algorithms share the same model, and we'd like to warm-start one
    algorithm with the model of another. Some algorithms (e.g. FixMatch) save the featurizer, classifier within a sequential,
    and thus the featurizer keys may look like 'model.module.0._' 'model.0._' or 'model.module.model.0._',
    and the classifier keys may look like 'model.module.1._' 'model.1._' or 'model.module.model.1._'
    while simple algorithms (e.g. ERM) use no sequential 'model._'
    """
    # hard-coded exceptions
    d = {re.sub('model.1.', 'model.classifier.', k): v for k,v in d.items()}
    d = {k: v for k,v in d.items() if 'pre_classifier' not in k} # this causes errors

    # probe the proper transformation from d.keys() -> reference
    # do this by splitting d's first key on '.' until we get a string that is a strict substring of something in ref
    success = False
    probe = list(d.keys())[0].split('.')
    for i in range(len(probe)):
        probe_str = '.'.join(probe[i:])
        matches = list(filter(lambda ref_k: len(ref_k) >= len(probe_str) and probe_str == ref_k[-len(probe_str):], ref))
        matches = list(filter(lambda ref_k: not 'layer' in ref_k, matches)) # handle resnet probe being too simple, e.g. 'weight'
        if len(matches) == 0: continue
        else:
            success = True
            append = [m[:-len(probe_str)] for m in matches]
            remove = '.'.join(probe[:i]) + '.'
            break
    if not success: raise Exception("These dictionaries have irreconcilable keys")

    return_d = {}
    for a in append:
        for k,v in d.items(): return_d[re.sub(remove, a, k)] = v

    # hard-coded exceptions
    if 'model.classifier.weight' in return_d:
       return_d['model.1.weight'], return_d['model.1.bias'] = return_d['model.classifier.weight'], return_d['model.classifier.bias']
    return return_d

def log_group_data(datasets, grouper, logger):
    for k, dataset in datasets.items():
        name = dataset['name']
        dataset = dataset['dataset']
        logger.write(f'{name} data...\n')
        if grouper is None:
            logger.write(f'    n = {len(dataset)}\n')
        else:
            _, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_counts = group_counts.tolist()
            for group_idx in range(grouper.n_groups):
                logger.write(f'    {grouper.group_str(group_idx)}: n = {group_counts[group_idx]:.0f}\n')
    logger.flush()

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class BatchLogger:
    def __init__(self, csv_path, mode='w', use_wandb=False):
        self.path = csv_path
        self.mode = mode
        self.file = open(csv_path, mode)
        self.is_initialized = False

        # Use Weights and Biases for logging
        self.use_wandb = use_wandb
        if use_wandb:
            self.split = Path(csv_path).stem

    def setup(self, log_dict):
        columns = log_dict.keys()
        # Move epoch and batch to the front if in the log_dict
        for key in ['batch', 'epoch']:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if self.mode=='w' or (not os.path.exists(self.path)) or os.path.getsize(self.path)==0:
            self.writer.writeheader()
        self.is_initialized = True

    def log(self, log_dict):
        if self.is_initialized is False:
            self.setup(log_dict)
        self.writer.writerow(log_dict)
        self.flush()

        if self.use_wandb:
            results = {}
            for key in log_dict:
                new_key = f'{self.split}/{key}'
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_config(config, logger):
    for name, val in vars(config).items():
        logger.write(f'{name.replace("_"," ").capitalize()}: {val}\n')
    logger.write('\n')

def initialize_wandb(config):
    if config.wandb_api_key_path is not None:
        with open(config.wandb_api_key_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    wandb.init(name='_'.join(config.log_dir.split('/')[-2:]))
    wandb.config.update(config)

def save_pred(y_pred, path_prefix):
    # Single tensor
    if torch.is_tensor(y_pred):
        df = pd.DataFrame(y_pred.numpy())
        df.to_csv(path_prefix + '.csv', index=False, header=False)
    # Dictionary
    elif isinstance(y_pred, dict) or isinstance(y_pred, list):
        torch.save(y_pred, path_prefix + '.pth')
    else:
        raise TypeError("Invalid type for save_pred")

def get_replicate_str(dataset, config):
    if dataset['dataset'].dataset_name == 'poverty':
        replicate_str = f"fold:{config.dataset_kwargs['fold']}"
    else:
        replicate_str = f"seed:{config.seed}"
    return replicate_str

def get_pred_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    split = dataset['split']
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_split:{split}_{replicate_str}_")
    return prefix

def get_model_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_{replicate_str}_")
    return prefix

def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)

def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")

def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.
    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.
    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  make_pipeline
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
import torch
from torch.utils.data import Dataset


# records partition function values
def get_noise_in_estimator(n, n_noise, dataset):
    # rescale the partition funciton / Z by n_noise and the data distribution
    if dataset == "mnist":
        # with EE
        noise_in_estimator_tsne = 8.13 * 10**6

        # without EE
        #noise_in_estimator_tsne = 6.25 * 10**6

        # using Z not norm for NCVis
        noise_in_estimator_ncvis = 3.43 * 10**7

    elif dataset == "human-409b2":
        noise_in_estimator_tsne = 1.30 * 10**6
        noise_in_estimator_ncvis = 3.57 * 10**6 # using Z not norm

    elif dataset == "imba_mnist_odd_seed_0":
        noise_in_estimator_tsne = 3.16 * 10**6
        noise_in_estimator_ncvis = 7.88 * 10**6

    elif dataset == "imba_mnist_lin_seed_0":
        noise_in_estimator_tsne = 3.12 * 10**6
        noise_in_estimator_ncvis = 6.15 * 10**6
    elif dataset == "zebrafish":
        noise_in_estimator_tsne = 7.98 * 10**6
        noise_in_estimator_ncvis = 3.08 * 10**7
    elif dataset == "c_elegans":
        noise_in_estimator_tsne = 1.17 * 10**7
        noise_in_estimator_ncvis = 3.69 * 10**7
    elif dataset == "k49":
        noise_in_estimator_tsne = 7.98 * 10**6
        noise_in_estimator_ncvis = 3.95 * 10**8
    else:
        noise_in_estimator_tsne = 7.98 * 10**6
        noise_in_estimator_ncvis = 3.08 * 10**7
    noise_in_estimator_tsne = noise_in_estimator_tsne * n_noise / n / (n-1)
    noise_in_estimator_ncvis = noise_in_estimator_ncvis * n_noise / n / (n-1)


    noise_in_estimator_tsne = float(np.format_float_scientific(noise_in_estimator_tsne,
                                                           precision=2))
    noise_in_estimator_ncvis = float(np.format_float_scientific(noise_in_estimator_ncvis,
                                                            precision=2))

    nbs_noise_in_estimator =  np.array([noise_in_estimator_tsne,
                                        noise_in_estimator_ncvis,
                                        5e-5,
                                        1e-4,
                                        2e-4, 5e-4,
                                        1e-3,
                                        2e-3, 5e-3,
                                        1e-2,
                                        2e-2, 5e-2,
                                        1e-1,
                                        2e-1,
                                        5e-1,
                                        1e-0, 2e-0, 5e-0,
                                        1e1, 2e1, 5e1,
                                        1e2, 2e2, 5e2
                                        ])

    return nbs_noise_in_estimator


# helper for path management
def get_path(path_type):
    with open("../my_paths", "r") as file:
        lines = file.readlines()

    lines = [line.split(" ") for line in lines]
    path_dict = {line[0]: line[1].strip("\n") for line in lines}

    if path_type == "data":
        try:
            return path_dict["data_path"]
        except KeyError:
            print("There is no path 'data_path'.")

    elif path_type == "figures":
        try:
            return path_dict["fig_path"]
        except KeyError:
            print("There is no path 'fig_path'.")


# CIFAR-specific numbers
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
dim = (32, 32, 3)


# evaluation code for SimCLR model

# data loader for cifar
class TransformLabelData(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        labels,
        mean,
        std,
        size,
        reshape=None,
        setting="train",
    ):
        self.dataset = dataset
        self.labels = labels
        if self.dataset.max() <= 1.0:
            self.dataset *= 255
        self.dataset = self.dataset.astype(np.uint8)

        if reshape is not None:
            # lambda d: torch.reshape(d, (-1, 28, 28))
            self.dataset = reshape(self.dataset)
        else:
            self.dataset = np.reshape(self.dataset, (-1, *size))

        self.transform = get_transforms(mean, std, size[:2], setting=setting)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i, :, :]

        return self.transform(item), self.labels[i]


# get data loaders for evaluation
def get_features(data,
                 labels,
                 embedder,
                 run=0,
                 test_size=10000,
                 batch_size=1024,
                 num_workers=8,
                 device="cuda:0"
                 ):
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        random_state=run,
        test_size=test_size,
        stratify=labels,
    )

    train_dataset = TransformLabelData(
    X_train, y_train, mean, std, dim, setting="test"  # we are not using any augmentation for training or testing the classifiers
    )

    gen = torch.Generator().manual_seed(run)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=gen,
        drop_last=False,
    )

    test_dataset = TransformLabelData(X_test, y_test, mean, std, dim, setting="test")
    gen = torch.Generator().manual_seed(run)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=gen,
        drop_last=False,
    )

    model = embedder.model.resnet
    model = model.eval()
    with torch.no_grad():
        features_train = np.vstack(
            [
                model(b.to(device)).cpu().detach().numpy()
                for b, _ in train_loader
            ]
        )

    with torch.no_grad():
        features_test = np.vstack(
            [
                model(b.to(device)).cpu().detach().numpy()
                for b, _ in test_loader
            ]
        )

    return features_train, features_test, y_train, y_test


# compute the knn acc
def compute_knn_acc(data,
                    labels,
                    embedder,
                    run=0,
                    test_size=10000,
                    batch_size=1024,
                    num_workers=8,
                    device="cuda:0"
                    ):

    features_train, features_test, y_train, y_test = get_features(data,
                                                                  labels,
                                                                  embedder,
                                                                  run,
                                                                  test_size,
                                                                  batch_size,
                                                                  num_workers,
                                                                  device)

    knn = KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1)
    knn.fit(features_train, y_train)
    acc = knn.score(features_test, y_test)
    return acc


# wrapper for knn accuracy computation
def check_knn_acc(data,
                  labels,
                  embedder,
                  file_name,
                  run=0,
                  test_size=10000,
                  batch_size=1024,
                  num_workers=8,
                  device="cuda:0"
                  ):

    if hasattr(embedder, 'knn_acc'):
        return embedder.knn_acc
    else:
        acc = compute_knn_acc(data,
                              labels,
                              embedder,
                              run,
                              test_size,
                              batch_size,
                              num_workers,
                              device)

        embedder.knn_acc = acc

        # save model with knn acc
        with open(file_name, "wb") as file:
            pickle.dump(embedder, file, pickle.HIGHEST_PROTOCOL)
        return acc


# compute linear accuracy with sklearn
def compute_lin_acc(data,
                    labels,
                    embedder,
                    run=0,
                    test_size=10000,
                    batch_size=1024,
                    num_workers=8,
                    device="cuda:0"):

    features_train, features_test, y_train, y_test = get_features(data,
                                                                  labels,
                                                                  embedder,
                                                                  run,
                                                                  test_size,
                                                                  batch_size,
                                                                  num_workers,
                                                                  device)
    clf = make_pipeline(
      StandardScaler(),
      LogisticRegression(
          penalty="none", solver="saga", tol=1e-4, random_state=run, n_jobs=-1, max_iter=1000
      ),
    )
    clf.fit(features_train, y_train)
    acc = clf.score(features_test, y_test)
    return acc


# wrapper for linear accurarcy computation
def check_lin_acc(data,
                  labels,
                  embedder,
                  file_name,
                  run=0,
                  test_size=10000,
                  batch_size=1024,
                  num_workers=8,
                  device="cuda:0"
                  ):

    if hasattr(embedder, 'lin_acc'):
        return embedder.lin_acc
    else:
        acc = compute_lin_acc(data,
                              labels,
                              embedder,
                              run,
                              test_size,
                              batch_size,
                              num_workers,
                              device)

        embedder.lin_acc = acc

        # save model with knn acc
        with open(file_name, "wb") as file:
            pickle.dump(embedder, file, pickle.HIGHEST_PROTOCOL)
        return acc


# dataloaders for network training
def get_transforms(mean, std, size, setting="train"):
    normalize = transforms.Normalize(mean=mean, std=std)

    if setting == "full":
        transform = transforms.Compose(
            [
                transforms.functional.to_pil_image,
                # transforms.RandomRotation(30),
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.functional.to_tensor,
                normalize,
            ]
        )

    elif setting == "train":
        transform = transforms.Compose(
            [
                transforms.functional.to_pil_image,
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif setting == "test":
        transform = transforms.Compose(
            [
                transforms.functional.to_pil_image,
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(f"Unknown transform {setting = }")

    return transform


class DistortionTransformData(Dataset):
    """Returns a pair of neighboring points in the dataset."""

    def __init__(self, dataset, mean, std, size, reshape=None):
        self.dataset = dataset
        if self.dataset.max() <= 1.0:
            self.dataset *= 255
        self.dataset = self.dataset.astype(np.uint8)

        if reshape is not None:
            # lambda d: torch.reshape(d, (-1, 28, 28))
            self.dataset = reshape(self.dataset)
        else:
            self.dataset = np.reshape(self.dataset, (-1, *size))

        self.transform = get_transforms(mean, std, size[:2], setting="full")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i, :, :]

        return self.transform(item), self.transform(item)


# network architecture
class FCNetwork(nn.Module):
    "Fully-connected network"

    def __init__(self, in_dim=784, feat_dim=128, hidden_dim=1024):
        super(FCNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ResNetFC(nn.Module):
    def __init__(
        self,
        model_name="resnet18",
        in_channel=3,
        feat_dim=2,
        hidden_dim=1024,
        normalize=False,
    ):
        super(ResNetFC, self).__init__()
        if model_name == "resnet18":
            self.resnet = resnet18()
            in_dim = 512
        elif model_name == "resnet50":
            self.resnet = torchvision.models.resnet50()
            in_dim = 2048
        self.feat_dim = feat_dim
        self.backbone_dim = in_dim

        self.fc = FCNetwork(in_dim=in_dim, feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.normalize = normalize

    def forward(self, x):
        h = self.resnet(x)
        z = self.fc(h)
        if self.normalize:
            z = F.normalize(z, dim=1)
        return z


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)