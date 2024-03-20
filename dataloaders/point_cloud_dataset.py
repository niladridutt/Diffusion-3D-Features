from utils import str2bool
import os
import os.path
import numpy as np
from utils import to_numpy, to_tensor
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch_cluster import knn


np.random.seed(42)

def get_max_dist(point_cloud):
    """
    Computes the maximal Euclidean distance between points in the point cloud
    Args:
        point_cloud: numpy array with shape n x 3

    Returns:
        d_max: a float32 value
    """

    expanded_row = np.expand_dims(point_cloud, axis=0)
    expanded_col = np.expand_dims(point_cloud, axis=1)

    d2_mat = np.sum((expanded_row - expanded_col) ** 2, axis=2)
    d_max = np.sqrt(d2_mat.max())

    return d_max


def matrix_map_from_corr_map(matrix_map, source, target):
    # matrix_map.shape is [N,], output.shape is [N,M]
    matrix_map = matrix_map.clone().detach().to(source.device)
    inputs = torch.ones((source.shape[0], target.shape[0])).to(matrix_map.device)
    outputs = torch.zeros((source.shape[0], target.shape[0])).to(matrix_map.device)

    label = outputs.scatter(dim=1, index=matrix_map.unsqueeze(1).long(), src=inputs)
    return label


class PointCloudDataset(Dataset):
    def __init__(self, params, split="train"):
        self.hparams = params
        self.split = split
        self.data_root = f"data/datasets/{params.dataset_name}"
        self.verts, self.faces, self.d_max, self.gt_map = self.load_data(
            self.data_root, split, self.hparams
        )
        self.pair_indices = self.valid_pairs(self.gt_map)


    def __getitem__(self, index):
        from dataloaders.tosca import TOSCA
        from dataloaders.shrec_20 import SHREC20
        from dataloaders.shrec07 import SHREC07
        from dataloaders.snis import SNIS
        pair_str = "_".join([str(i) for i in self.pair_indices[index]])
        id1, id2 = self.pair_indices[index % len(self.pair_indices)]
        
        if (
            self.hparams.OVERFIT_singel_pair is not None
            and self.hparams.OVERFIT_singel_pair is not False
        ):
            id1, id2 = [int(id) for id in self.hparams.OVERFIT_singel_pair.split("_")]

        X_verts = to_numpy(self.verts[id1])
        Y_verts = to_numpy(self.verts[id2])
        if isinstance(self, TOSCA):
            X_verts = X_verts / 10
            Y_verts = Y_verts / 10

        X, Y = {"pos": X_verts}, {"pos": Y_verts}
        X["id"], Y["id"] = id1, id2

        X["d_max"], Y["d_max"] = (
            torch.tensor(
                [
                    self.d_max[id1],
                ]
            ).float(),
            torch.tensor(
                [
                    self.d_max[id2],
                ]
            ).float(),
        )

        min_num_points = (
            X["pos"].shape[0]
            if X["pos"].shape[0] < Y["pos"].shape[0]
            else Y["pos"].shape[0]
        )
        gt_map = torch.arange(X["pos"].shape[0])

        if min_num_points > self.hparams.num_points:
            Y["rand_choice"] = X["rand_choice"] = np.random.choice(
                min_num_points, self.hparams.num_points, replace=False
            )
            X["pos"] = X_verts[X["rand_choice"]]
            Y["pos"] = Y_verts[X["rand_choice"]]
            if self.gt_map is None:
                pass
            elif isinstance(self, SHREC20) or isinstance(self, SNIS) or isinstance(self, SHREC07):
                full = False # test for full or partial shrec20
                s = self.gt_map[pair_str][:,0].reshape(-1,1)
                t = self.gt_map[pair_str][:,1].reshape(-1,1)
                if full:
                    choices = np.arange(X_verts.shape[0])
                    choices[s[:,0]] = -1
                    choices = choices[choices!=-1]
                    X["rand_choice"] = np.random.choice(
                    choices, self.hparams.num_points-s.shape[0], replace=False
                    )
                    X["rand_choice"]  = np.hstack([s[:,0],X["rand_choice"]])
                    choices = np.arange(Y_verts.shape[0])
                    choices[t[:,0]] = -1
                    choices = choices[choices!=-1]
                    Y["rand_choice"] = np.random.choice(
                    choices, self.hparams.num_points-t.shape[0], replace=False
                    )
                    Y["rand_choice"]  = np.hstack([t[:,0],Y["rand_choice"]])
                    X["pos"] = X_verts[X["rand_choice"]]
                    Y["pos"] = Y_verts[Y["rand_choice"]]
                    gt_map = np.arange(Y_verts.shape[0])
                    Y["map_start_index"] = t.shape[0]
                else:
                    X["rand_choice"] = s.squeeze()
                    Y["rand_choice"] = t.squeeze()
                    X["pos"] = X_verts[X["rand_choice"]]
                    Y["pos"] = Y_verts[Y["rand_choice"]]
                    gt_map = np.arange(s.shape[0])
                    Y["map_start_index"] = t.shape[0]
            elif isinstance(self.gt_map, np.ndarray) and np.allclose(
                self.gt_map[index], np.arange(self.gt_map[index].shape[0])
            ):
                gt_map = np.arange(Y_verts.shape[0])
            elif (
                pair_str in self.gt_map
                and self.gt_map[pair_str] is not None
                and isinstance(self.gt_map, dict)
            ):
                gt_map = self.gt_map[pair_str][Y["rand_choice"]]
                Y["pos"] = Y_verts[gt_map]
                Y["rand_choice"] = np.random.choice(
                    self.hparams.num_points, self.hparams.num_points, replace=False
                )
                Y["org"] = gt_map[Y["rand_choice"]]
                Y["pos"] = Y["pos"][Y["rand_choice"]]
                gt_map = np.argsort(Y["rand_choice"])
            else:
                gt_map = np.arange(Y_verts.shape[0])

        for pc in [X, Y]:
            for k, v in pc.items():
                if isinstance(v, np.ndarray):
                    pc[k] = torch.from_numpy(v.astype("float32"))

            pc["edge_index"] = knn(
                pc["pos"], pc["pos"], k=self.hparams.num_neighs, num_workers=20
            )
            pc["neigh_idxs"] = pc["edge_index"][1].reshape(pc["pos"].shape[0], -1)

        data = {"source": X, "target": Y}
        if self.gt_map is not None:
            data["gt_map"] = gt_map
        if (
            to_tensor(gt_map).allclose(torch.arange(gt_map.shape[0]))
            and Y["pos"].shape[0] < gt_map.shape[0]
        ):
            data["gt_map"] = torch.arange(Y["pos"].shape[0])

        return data

    def __len__(self):
        return len(self.pair_indices)

    @staticmethod
    def add_dataset_specific_args(
        parser, task_name, dataset_name, is_lowest_leaf=False
    ):
        parser.add_argument(
            "--num_points",
            type=int,
            default=1024,
            help="Number of points in point cloud",
        )

        parser.add_argument("--split", default="train", help="Which data to fetch")

        parser.add_argument(
            "--num_neighs", type=int, default=40, help="Num of nearest neighbors to use"
        )
        parser.add_argument(
            "--tosca_all_pairs", nargs="?", default=False, type=str2bool, const=True
        )
        parser.add_argument(
            "--tosca_cross_pairs", nargs="?", default=False, type=str2bool, const=True
        )

        if parser.parse_known_args()[0].dataset_name in ["surreal", "smal"]:
            parser.set_defaults(limit_train_batches=1000, limit_val_batches=200)
        return parser

    @staticmethod
    def extract_soft_labels_per_pair(gt_map, target_pc, replace_on_cpu=False):
        ratio_list = (0.002 * np.arange(1, 101)).tolist()
        soft_labels = {}
        for each_ratio in ratio_list:
            val = PointCloudDataset.make_soft_label(
                gt_map, target_pc, ratio=each_ratio
            )  # torch.Size([1024, 1024])
            if replace_on_cpu:
                val = val.cpu()
            soft_labels[f"{each_ratio}"] = val
        return ratio_list, soft_labels

    @staticmethod
    def make_soft_label(label_origin, xyz2, ratio=0.5):
        if ratio == 0.0:
            return label_origin
        else:
            soft_label = label_origin.clone()

            dist = torch.cdist(xyz2, xyz2) ** 2

            max_square_radius = torch.max(dist)

            radius = ratio * torch.sqrt(max_square_radius)

            dists_from_source = dist[soft_label.nonzero(as_tuple=False)[:, 1]]
            mask = dists_from_source <= radius**2
            soft_label[: mask.shape[0]][mask] = 1
            return soft_label

    @staticmethod
    def compute_or_load_soft_labels(indices_pairs, verts, gts, data_path):
        if os.path.exists(f"{data_path}/soft_labels"):
            return torch.load(f"{data_path}/soft_labels")

        soft_labels = {}
        for idx, (source_idx, target_idx) in enumerate(
            tqdm(indices_pairs, desc="compute_soft_labels")
        ):
            input1 = torch.tensor(verts[source_idx])
            input2 = torch.tensor(verts[target_idx])
            gt_map = gts[idx]
            ratio_list, soft_labels = PointCloudDataset.extract_soft_labels_per_pair(
                source_idx, target_idx, gt_map, input1, input2
            )

        torch.save(soft_labels, f"{data_path}/soft_labels")
        return soft_labels


if __name__ == "__main__":
    pass
