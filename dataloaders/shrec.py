#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataloaders.point_cloud_dataset import PointCloudDataset, get_max_dist
import os
from pathlib import Path
import sys
from dataloaders.mesh_container import MeshContainer



import numpy as np
import torch
import itertools


from tqdm import tqdm



def download(shrec_data_path):
    os.makedirs(shrec_data_path, exist_ok=True)
    unified_path = os.path.join(shrec_data_path, 'unified')
    if not os.path.exists(unified_path):
        zip_path = os.path.join(shrec_data_path, 'SHREC_r.zip')
        download_command = f"wget --header=\"Host: nuage.lix.polytechnique.fr\" --header=\"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36\" --header=\"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9\" --header=\"Accept-Language: en-US,en;q=0.9,he-IL;q=0.8,he;q=0.7\" --header=\"Cookie: __utma=146708137.1725280282.1622015292.1622015292.1622015292.1; __utmz=146708137.1622015292.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); oc_sessionPassphrase=UBol3DP5eEbaGWjMpBBerNqPdCAXkZmVUyEnDvJj0Aw%2F2Otbzhxvy9Jaq26v9T1BcCR7cZQWZeroDfZcWS5qJaS%2FZsCL2DGbOnnZtCrYkwd2%2B1A4jXbxKMIfOooHIBBC; occulifztxc2=fsjspr4igecrj3el6jqqeonr0d; __Host-nc_sameSiteCookielax=true; __Host-nc_sameSiteCookiestrict=true\" --header=\"Connection: keep-alive\" \"https://nuage.lix.polytechnique.fr/index.php/s/LJFXrsTG22wYCXx/download?path=%2F&files=SHREC_r.zip\" -c -O {zip_path}"
        apply_download(download_command, zip_path, shrec_data_path)

        zip_path = zip_path.replace('_r','_r_gt')
        download_command = f"wget --header=\"Host: nuage.lix.polytechnique.fr\" --header=\"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36\" --header=\"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9\" --header=\"Accept-Language: en-US,en;q=0.9,he-IL;q=0.8,he;q=0.7\" --header=\"Cookie: __utma=146708137.1725280282.1622015292.1622015292.1622015292.1; __utmz=146708137.1622015292.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); oc_sessionPassphrase=UBol3DP5eEbaGWjMpBBerNqPdCAXkZmVUyEnDvJj0Aw%2F2Otbzhxvy9Jaq26v9T1BcCR7cZQWZeroDfZcWS5qJaS%2FZsCL2DGbOnnZtCrYkwd2%2B1A4jXbxKMIfOooHIBBC; occulifztxc2=fsjspr4igecrj3el6jqqeonr0d; __Host-nc_sameSiteCookielax=true; __Host-nc_sameSiteCookiestrict=true\" --header=\"Connection: keep-alive\" \"https://nuage.lix.polytechnique.fr/index.php/s/LJFXrsTG22wYCXx/download?path=%2F&files=SHREC_r_gt.zip\" -c -O {zip_path}"
        apply_download(download_command, zip_path, shrec_data_path)


def apply_download(download_command, download_path, destination_path):
    os.system(f'{download_command}')
    os.system(f'unzip {download_path} -d {destination_path}')
    os.system(f'rm {download_path}')


class SHREC(PointCloudDataset):

    def __init__(self, hparams,split):
        super(SHREC, self).__init__(hparams,split=split)
        if(self.split == 'train'):
            self.gt_map = None
    
    
    def valid_pairs(self,gt_map):
        if(self.split == 'test'):
            return [[int(idx) for idx in k.split('_')] for k in list(gt_map.keys())]
        else:
            pairs = list(itertools.product(list(range(len(self.verts))), list(range(len(self.verts)))))
            return list(filter(lambda pair: pair[0] != pair[1],pairs))
    
    def __getitem__(self, item):
        out_dict = super(SHREC, self).__getitem__(item)
        return out_dict


    @staticmethod
    def add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = PointCloudDataset.add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf)
        parser.set_defaults(test_on_shrec=True)
        return parser

    @staticmethod
    def load_data(*args):
        shrec_data_path = 'datasets/shrec'
        if not os.path.exists(f"{shrec_data_path}/off_2"):
            download(shrec_data_path)
        shapes_path = f"{shrec_data_path}/off_2"
        gt_path = f"{shrec_data_path}/groundtruth"
        if(not os.path.exists(f"{shrec_data_path}/unified")):
            all_verts,all_faces,all_d_max,all_maps = [],[],[],{}
            sorted_paths = sorted([str(path) for path in list(Path(shapes_path).rglob("*.off"))],key=lambda p:int(os.path.basename(p)[:-4]))
            sorted_gt_paths = sorted([str(path) for path in list(Path(gt_path).rglob("*.map"))],key=lambda p:int(os.path.basename(p)[:-4].split('_')[0]))

            for off in tqdm(sorted_paths, desc="Unifying shrec"):
                mesh = MeshContainer().load_from_file(str(off))
                all_verts.append(mesh.vert)
                all_faces.append(mesh.face)
                all_d_max.append(get_max_dist(mesh.vert))
            for map in tqdm(sorted_gt_paths, desc="Unifying shrec maps"):
                idxs = [int(i)-1 for i in os.path.basename(map)[:-4].split("_")]
                idxs_string = "_".join([str(i) for i in idxs])
                all_maps[idxs_string] = np.loadtxt(map) - 1 # 1-indexed
            torch.save((all_verts,all_faces,all_d_max,all_maps),f"{shrec_data_path}/unified")
        else:
            all_verts,all_faces,all_d_max,all_maps = torch.load(f"{shrec_data_path}/unified")
            all_maps = {k:v.astype(np.int64) for k,v in all_maps.items()}
        

        return all_verts,all_faces,all_d_max,all_maps
