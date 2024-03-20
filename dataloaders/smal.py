from dataloaders.surreal import Fake_pair_indices
from dataloaders.point_cloud_dataset import PointCloudDataset, get_max_dist
import itertools
import os
import os.path
import numpy as np
import sys

from tqdm.auto import tqdm
from dataloaders.mesh_container import MeshContainer
import math
import torch
from pathlib import Path
from dataloaders.mesh_container import MeshContainer
from tqdm import tqdm





def prepare_smal_original_data(split,hparams):
    smal_original_data_path = 'data/datasets/smal/generated_samples'
    if not os.path.exists(smal_original_data_path):
        smal_original_data_path = os.path.join('~', 'data', 'datasets', 'smal', 'generated_samples')

    data_size = -1
    # if(hparams.train_on_limited_data and split != 'test'):
    #     data_size = int(hparams.train_on_limited_data * (0.9 if split=='train' else 0.1))
    split = 'val' if split == 'test' else 'train'
    data_path = f"{smal_original_data_path}/data_{split}.pth"
    if(not os.path.exists(data_path)):
        # split_for_orig = 'train' if split != 'test' else 'test'
        verts = []
        for file in tqdm(list(Path(smal_original_data_path).rglob(f"*{split}*"))):
            shape = MeshContainer().load_from_file(str(file))
            verts.append(shape.vert)
        face = shape.face
        torch.save((verts,face),data_path,pickle_protocol=4)
    return data_path



class SMAL(PointCloudDataset):
    def __init__(self, params, split='train'):
        super(SMAL, self).__init__(params,split)

    @staticmethod
    def add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = PointCloudDataset.add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False)
        parser.set_defaults(limit_train_batches=5000,limit_val_batches=200,limit_test_batches=1000)
        return parser
    
    
    def valid_pairs(self,gt_map):
        if(self.split == 'test'):
            gt_pairs_path = 'data/datasets/smal/generated_samples/gt_pairs_indices'
            if not os.path.exists(gt_pairs_path):
                gt_pairs_path = os.path.join('.', 'data', 'datasets', 'smal', 'generated_samples', 'gt_pairs_indices')
            if(not os.path.exists(gt_pairs_path)):
                num_shapes = int(math.sqrt(gt_map.shape[0]))
                all_pairs = list(itertools.product(list(range(num_shapes)), list(range(num_shapes))))
                all_pairs = np.array(list(filter(lambda pair: pair[0] != pair[1],all_pairs)))
                indices = np.random.choice(len(all_pairs), int(self.hparams.limit_test_batches), replace=False).tolist()
                all_pairs = all_pairs[indices]
                torch.save(all_pairs,gt_pairs_path)
            all_pairs = torch.load(gt_pairs_path)
            return all_pairs
        else:
            return Fake_pair_indices(int(math.sqrt(gt_map.shape[0])))
        

    @staticmethod
    def load_data(data_root, split,hparams):
        data_path = prepare_smal_original_data(split,hparams)
        datas,face = torch.load(data_path)
        # datas = datas[-int(len(datas) * 0.1):] if split == 'val' else datas[:int(len(datas) * 0.9)] 
        d_max = np.array([get_max_dist(datas[0])])
        faces_repeated = np.broadcast_to(face[None,:], (len(datas), *face.shape))
        all_d_max = np.broadcast_to(d_max[None,:], (len(datas), *d_max.shape))[:,0]

        # The gt map for smal_ORIGINAL is all for all
        gt_is_eye =  np.broadcast_to(np.arange(datas[0].shape[0])[None,:], (len(datas) ** 2, datas[0].shape[0]))
        return datas,faces_repeated,all_d_max, gt_is_eye

if __name__ == "__main__":
    smal_ORIGINAL.load_data(data_root='',split='train')
