#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from dataloaders.shrec import apply_download
from dataloaders.point_cloud_dataset import PointCloudDataset, get_max_dist
import os
import re
from pathlib import Path
import sys
from dataloaders.mesh_container import MeshContainer
import numpy as np
import torch
import itertools
from tqdm import tqdm


def download(tosca_data_path):
    os.makedirs(tosca_data_path, exist_ok=True)
    unified_path = os.path.join(tosca_data_path, "unified")
    if not os.path.exists(unified_path):
        zip_path = os.path.join(tosca_data_path, "toscahires-mat.zip")
        download_command = f'wget --header="Host: tosca.cs.technion.ac.il" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,he-IL;q=0.8,he;q=0.7" --header="Referer: http://tosca.cs.technion.ac.il/book/resources_data.html" --header="Cookie: _ga=GA1.3.161556134.1618078345; __utma=132748060.161556134.1618078345.1624638699.1624638699.1; __utmz=132748060.1624638699.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utmc=65548203; __utma=65548203.161556134.1618078345.1625382417.1625396898.2; __utmz=65548203.1625396898.2.2.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utmt=1; __utmb=65548203.1.10.1625396898; sc_is_visitor_unique=rx2635946.1625396909.5C7B5D1FE92F4F57BC9A5E1EAE78D6C4.2.2.2.2.1.1.1.1.1" --header="Connection: keep-alive" "http://tosca.cs.technion.ac.il/data/toscahires-mat.zip" -c -O {zip_path}'
        apply_download(download_command, zip_path, tosca_data_path)


class TOSCA(PointCloudDataset):
    def __init__(self, hparams, split):
        super(TOSCA, self).__init__(hparams, split=split)

    def valid_pairs(self, gt_map):
        if self.hparams.tosca_all_pairs:
            pairs = list(
                itertools.product(
                    list(range(len(self.verts))), list(range(len(self.verts)))
                )
            )
            return list(filter(lambda pair: pair[0] != pair[1], pairs))
        elif self.hparams.tosca_cross_pairs:
            all = list(
                itertools.product(
                    list(range(len(self.verts))), list(range(len(self.verts)))
                )
            )
            all = list(filter(lambda pair: pair[0] != pair[1], all))
            intra = [
                tuple(int(idx) for idx in k.split("_")) for k in list(gt_map.keys())
            ]
            only_cross = [elem for elem in all if elem not in intra]
            return only_cross
        else:
            return [[int(idx) for idx in k.split("_")] for k in list(gt_map.keys())]

    def __getitem__(self, item):
        out_dict = super(TOSCA, self).__getitem__(item)
        return out_dict

    @staticmethod
    def add_dataset_specific_args(
        parser, task_name, dataset_name, is_lowest_leaf=False
    ):
        parser = PointCloudDataset.add_dataset_specific_args(
            parser, task_name, dataset_name, is_lowest_leaf
        )
        parser.set_defaults(test_on_tosca=True)
        return parser

    @staticmethod
    def load_data(*args):
        tosca_data_path = "datasets/tosca/Meshes"
        if not os.path.exists(f"{tosca_data_path}/unified"):
            all_verts, all_faces, all_d_max, all_maps = [], [], [], {}
            sorted_paths = sorted(
                [str(path) for path in list(Path(tosca_data_path).rglob("*.off"))],
                key=lambda p: (
                    re.match(r"([a-z]+)([0-9]+)", os.path.basename(p), re.I).groups()[
                        0
                    ],
                    int(
                        re.match(
                            r"([a-z]+)([0-9]+)", os.path.basename(p), re.I
                        ).groups()[1]
                    ),
                ),
            )
            sorted_paths = list(filter(lambda file: 'victoria' not in file and 'david' not in file and 'michael' not in file,sorted_paths))

            for mat in tqdm(sorted_paths, desc="Unifying tosca"):
                mesh = MeshContainer().load_from_file(str(mat), dataset="tosca")
                all_verts.append(mesh.vert)
                all_faces.append(mesh.face)
                all_d_max.append(
                    torch.cdist(
                        torch.from_numpy(mesh.vert), torch.from_numpy(mesh.vert)
                    ).max()
                )
            gt_names = defaultdict(list)
            for i, mat in enumerate(tqdm(sorted_paths, desc="Unifying gt")):
                name, idx = re.match(
                    r"([a-z]+)([0-9]+)", os.path.basename(mat), re.I
                ).groups()
                gt_names[name].append(i)
            all_gt = {}
            for k in gt_names:
                all_pairs = list(itertools.product(gt_names[k], list(gt_names[k])))
                all_pairs = list(filter(lambda pair: pair[0] != pair[1], all_pairs))
                for p in all_pairs:
                    all_gt[p] = None
            torch.save(
                (all_verts, all_faces, all_d_max, all_gt), f"{tosca_data_path}/unified"
            )
        else:
            all_verts, all_faces, all_d_max, all_gt = torch.load(
                f"{tosca_data_path}/unified"
            )
        all_gt = {f"{k[0]}_{k[1]}": v for k, v in all_gt.items()}

        return all_verts, all_faces, all_d_max, all_gt
