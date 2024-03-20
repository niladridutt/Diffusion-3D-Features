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
import glob
from tqdm import tqdm
import scipy.io
import pandas as pd
from itertools import combinations


class SHREC07(PointCloudDataset):
    def __init__(self, hparams, split):
        super(SHREC07, self).__init__(hparams, split=split)
        if self.split == "train":
            self.gt_map = None

    def valid_pairs(self, gt_map):
        if self.split == "test":
            return [[int(idx) for idx in k.split("_")] for k in list(gt_map.keys())]
        else:
            return []

    def __getitem__(self, item):
        out_dict = super(SHREC07, self).__getitem__(item)
        return out_dict

    @staticmethod
    def add_dataset_specific_args(
        parser, task_name, dataset_name, is_lowest_leaf=False
    ):
        parser = PointCloudDataset.add_dataset_specific_args(
            parser, task_name, dataset_name, is_lowest_leaf
        )
        parser.set_defaults(test_on_shrec=True)
        return parser

    @staticmethod
    def load_data(*args):
        data_path = "datasets/CorrsBenchmark/Data/watertight_shrec07"
        shapes_path = f"{data_path}/Meshes"
        gt_path = f"{data_path}/Corrs"
        if not os.path.exists(f"{data_path}/unified"):
            all_verts, all_faces, all_d_max, all_maps = [], [], [], {}
            sorted_paths = sorted([str(path) for path in list(Path(shapes_path).rglob("*.off"))],key=lambda p:int(os.path.basename(p)[:-4]))
            filenames = [file.split("/")[-1].split(".")[0] for file in sorted_paths]
            for obj in tqdm(sorted_paths, desc="Unifying SHREC'07 shapes"):
                mesh = MeshContainer().load_from_file(str(obj))
                all_verts.append(mesh.vert)
                all_faces.append(mesh.face)
                all_d_max.append(get_max_dist(mesh.vert))
            
            # numbers = range(0, 20)
            numbers = range(20, 360)
            all_combinations = []
            step = 20

            for i in range(0, len(numbers), step):
                # Exclude combinations in the range 260-280
                current_combinations = list(combinations(numbers[i:i + step], 2))
                
                # Filter out combinations in the undesired range
                filtered_combinations = [
                    combo for combo in current_combinations
                    if not (260 <= combo[0] <= 280 or 260 <= combo[1] <= 280)
                ]
                all_combinations.extend(filtered_combinations)
            for idxs in tqdm(all_combinations, desc="Unifying SHREC'07 maps"):
                idxs_string = "_".join([str(i) for i in idxs])
                s = np.loadtxt(f"{gt_path}/{filenames[idxs[0]]}.vts")[:,0].astype(np.int32) 
                t = np.loadtxt(f"{gt_path}/{filenames[idxs[1]]}.vts")[:,0].astype(np.int32)
                all_maps[idxs_string] = np.stack([s, t],axis=1)
            torch.save(
                (all_verts, all_faces, all_d_max, all_maps),
                f"{data_path}/unified",
            )
        else:
            all_verts, all_faces, all_d_max, all_maps = torch.load(
                f"{data_path}/unified"
            )
            all_maps = {k: v.astype(np.int64) for k, v in all_maps.items()}

        return all_verts, all_faces, all_d_max, all_maps
