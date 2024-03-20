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


class SHREC20(PointCloudDataset):
    def __init__(self, hparams, split):
        super(SHREC20, self).__init__(hparams, split=split)
        if self.split == "train":
            self.gt_map = None

    def valid_pairs(self, gt_map):
        if self.split == "test":
            return [[int(idx) for idx in k.split("_")] for k in list(gt_map.keys())]
        else:
            return []

    def __getitem__(self, item):
        out_dict = super(SHREC20, self).__getitem__(item)
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
        shrec_data_path = "datasets/shrec20/"
        shapes_path = f"{shrec_data_path}/SHREC20b_lores/models/*.off"
        gt_path = f"{shrec_data_path}/SHREC20_lores_gts/SHREC20b_lores_gts"
        test_pairs = [
            ["giraffe_a", "giraffe_b"],
            ["elephant_a", "elephant_b"],
            ["camel_a", "giraffe_a"],
            ["giraffe_b", "camel_a"],
            ["cow", "bison"],
            ["dog", "leopard"],
            ["pig", "leopard"],
            ["leopard", "cow"],
            ["dog", "pig"],
            ["cow", "dog"],
            ["camel_a", "cow"],
            ["dog", "camel_a"],
            ["rhino", "cow"],
            ["pig", "elephant_a"],
            ["bison", "elephant_b"],
            ["rhino", "elephant_a"],
            ["cow", "elephant_a"],
            ["elephant_a", "giraffe_a"],
            ["cow", "giraffe_b"],
            ["dog", "giraffe_a"],
        ]
        if not os.path.exists(f"{shrec_data_path}/unified"):
            all_verts, all_faces, all_d_max, all_maps = [], [], [], {}
            sorted_paths = sorted(glob.glob(shapes_path))
            filenames = [file.split("/")[-1].split(".")[0] for file in sorted_paths]

            for off in tqdm(sorted_paths, desc="Unifying shrec20"):
                mesh = MeshContainer().load_from_file(str(off))
                all_verts.append(mesh.vert)
                all_faces.append(mesh.face)
                all_d_max.append(get_max_dist(mesh.vert))

            for pair in tqdm(test_pairs, desc="Unifying shrec20 maps"):
                idxs = [filenames.index(pair[0]), filenames.index(pair[1])]
                idxs_string = "_".join([str(i) for i in idxs])
                s = scipy.io.loadmat(f"{gt_path}/{filenames[idxs[0]]}.mat")
                t = scipy.io.loadmat(f"{gt_path}/{filenames[idxs[1]]}.mat")
                cp = np.intersect1d(s["points"], t["points"])
                s_idx = np.where(np.isin(s["points"], cp))[0]
                t_idx = np.where(np.isin(t["points"], cp))[0]
                s = s["verts"][s_idx] - 1
                t = t["verts"][t_idx] - 1
                all_maps[idxs_string] = np.hstack([s, t])
            torch.save(
                (all_verts, all_faces, all_d_max, all_maps),
                f"{shrec_data_path}/unified",
            )
        else:
            all_verts, all_faces, all_d_max, all_maps = torch.load(
                f"{shrec_data_path}/unified"
            )
            all_maps = {k: v.astype(np.int64) for k, v in all_maps.items()}

        return all_verts, all_faces, all_d_max, all_maps
