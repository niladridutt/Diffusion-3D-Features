from torch.utils.data.sampler import RandomSampler
import trimesh
from dataloaders.point_cloud_dataset import PointCloudDataset, get_max_dist

import os
import os.path
import numpy as np

import math
import torch

def download(path):
    try:
        os.makedirs(path,exist_ok=True)
        import gdown
        gdown.download("https://drive.google.com/u/0/uc?id=1VGax9j64AvCVORtiQ3ZSPecI0bfZrEVe",f"{path}/datas_surreal_test.pth")
        gdown.download("https://drive.google.com/u/0/uc?id=1HVReM43YtJqhGfbmE58dc1-edI_oz9YG",f"{path}/datas_surreal_train.pth")
    except:
        print("Failed to download")

def prepare_surreal_data(split,hparams):
    surreal_data_path = 'data/datasets/surreal'
    if not os.path.exists(surreal_data_path):
        download(surreal_data_path)
    data_size = -1
    if(hparams.train_on_limited_data and split != 'test'):
        data_size = int(hparams.train_on_limited_data * (0.9 if split=='train' else 0.1))
       
    data_path = f"{surreal_data_path}/datas_surreal_{split}{'' if (split=='test' or data_size < 0) else f'_{data_size}'}.pth"
    if(not os.path.exists(data_path)):
        split_for_orig = 'train' if split != 'test' else 'test'
        datas = torch.load(f"{surreal_data_path}/datas_surreal_{split_for_orig}.pth")
        if(hparams.train_on_limited_data is not None):
            datas = datas[:hparams.train_on_limited_data]
        datas = datas[-int(len(datas) * 0.1):] if split == 'val' else datas[:int(len(datas) * 0.9)] 
        torch.save(np.copy(datas),data_path,pickle_protocol=4)
    return data_path


class Fake_pair_indices:
    def __init__(self, len):
        self.length = len
    def __getitem__(self, key):
        source_idx = int(torch.div(key, (self.length - 1),rounding_mode='trunc'))
        # source_idx = key // (self.length - 1)
        target_idx = key % source_idx if source_idx > 0  else key
        target_idx = target_idx if target_idx < source_idx else target_idx + 1
        return source_idx,target_idx

    def __len__(self):
        return self.length ** 2 - self.length

class BigRandomSampler(RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        super(BigRandomSampler, self).__init__(data_source, replacement=True, num_samples=num_samples, generator=generator)
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        d_size = len(self.data_source)
        if(self.counter >= d_size):
            self.counter = 0
            raise StopIteration

        self.counter += 1
        return torch.randint(high=d_size, size=[1], dtype=torch.int64, generator=self.generator)[0]


    def __len__(self):
        return self.num_samples

class surreal(PointCloudDataset):
    def __init__(self, params, split='train'):
        super(surreal, self).__init__(params,split)

    @staticmethod
    def add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = PointCloudDataset.add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False)
        parser.set_defaults(limit_train_batches=5000,limit_val_batches=200,limit_test_batches=1000)
        return parser
    
    
    def valid_pairs(self,gt_map):
        # Fake all_pairs because too big
        return Fake_pair_indices(int(math.sqrt(gt_map.shape[0])))

    @staticmethod
    def load_data(data_root, split,hparams):
        data_path = prepare_surreal_data(split,hparams)
        datas = torch.load(data_path)

        d_max = np.array([get_max_dist(datas[0])])
        surreal_faces = trimesh.load("./data/surreal_template.ply", process=False).faces
        faces_repeated = np.broadcast_to(surreal_faces[None,:], (datas.shape[0], *surreal_faces.shape))
        all_d_max = np.broadcast_to(d_max[None,:], (datas.shape[0], *d_max.shape))[:,0]

        # The gt map for surreal is all for all
        gt_is_eye =  np.broadcast_to(np.arange(datas.shape[1])[None,:], (datas.shape[0] ** 2, datas.shape[1]))
        return datas,faces_repeated,all_d_max, gt_is_eye
