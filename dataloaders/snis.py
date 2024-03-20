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


class SNIS(PointCloudDataset):
    def __init__(self, hparams, split):
        super(SNIS, self).__init__(hparams, split=split)
        if self.split == "train":
            self.gt_map = None

    def valid_pairs(self, gt_map):
        if self.split == "test":
            return [[int(idx) for idx in k.split("_")] for k in list(gt_map.keys())]
        else:
            return []

    def __getitem__(self, item):
        out_dict = super(SNIS, self).__getitem__(item)
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
        snis_data_path = "datasets/SNIS"
        # shrec_data_path = '/cs/student/projects4/cgvi/2022/niladutt/SHREC_Rotated'
        smal_shapes_path = f"{snis_data_path}/gt_segmentation/smal/*.obj"
        faust_shapes_path = f"{snis_data_path}/gt_segmentation/faust/*.obj"
        gt_path = f"{snis_data_path}/keypoints_corres"
        test_pairs = pd.read_csv(f"{snis_data_path}/shape_pairs.csv")
        test_pairs = test_pairs[test_pairs["datasetA"] != "d4dt"]
        test_pairs = test_pairs[test_pairs["datasetB"] != "d4dt"]
        source, target = test_pairs["meshA"].to_list(), test_pairs["meshB"].to_list()
        test_pairs = []
        for i in range(len(source)):
            test_pairs.append([source[i].split(".")[0], target[i].split(".")[0]])
        if not os.path.exists(f"{snis_data_path}/unified"):
            all_verts, all_faces, all_d_max, all_maps = [], [], [], {}
            sorted_paths = sorted(glob.glob(smal_shapes_path) + glob.glob(faust_shapes_path))
            filenames = [file.split("/")[-1].split(".")[0] for file in sorted_paths]
            for obj in tqdm(sorted_paths, desc="Unifying SNIS shapes"):
                mesh = MeshContainer().load_from_file(str(obj))
                all_verts.append(mesh.vert)
                all_faces.append(mesh.face)
                all_d_max.append(get_max_dist(mesh.vert))
            extension = "_for_illustration"
            for pair in tqdm(test_pairs, desc="Unifying SNIS maps"):
                idxs = [filenames.index(pair[0]+extension), filenames.index(pair[1]+extension)]
                idxs_string = "_".join([str(i) for i in idxs])
                s = np.loadtxt(f"{gt_path}/{filenames[idxs[0]][:-17]}.vts").astype(np.int32) - 1
                t = np.loadtxt(f"{gt_path}/{filenames[idxs[1]][:-17]}.vts").astype(np.int32) - 1
                # s = all_verts[idxs[0]][s]
                # t = all_verts[idxs[1]][t]
                all_maps[idxs_string] = np.stack([s, t],axis=1)
            torch.save(
                (all_verts, all_faces, all_d_max, all_maps),
                f"{snis_data_path}/unified",
            )
        else:
            all_verts, all_faces, all_d_max, all_maps = torch.load(
                f"{snis_data_path}/unified"
            )
            all_maps = {k: v.astype(np.int64) for k, v in all_maps.items()}

        return all_verts, all_faces, all_d_max, all_maps
