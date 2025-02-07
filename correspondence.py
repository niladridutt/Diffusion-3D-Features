"""Shape correspondence template for testing from DPC"""
from argparse import Namespace
from torch.utils.data import DataLoader
from dataloaders.point_cloud_dataset import (
    PointCloudDataset,
    matrix_map_from_corr_map,
)
import torch
from torch import Tensor
from utils import to_numpy
from torchmetrics import Accuracy
import torch
import torch.nn.functional as F


class AccuracyAssumeEye(Accuracy):
    def __init__(self):
        super().__init__()

    def update(self, P: torch.Tensor, dim=1):
        preds = P.argmax(dim)

        dim_labels = dim - 1 if dim == len(P.shape) - 1 else dim + 1
        labels = (
            torch.arange(P.shape[dim_labels]).repeat(preds.shape[0], 1).to(preds.device)
        )
        super().update(preds, labels)


def square_distance(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src**2, -1).view(N, 1)
    dist += torch.sum(dst**2, -1).view(1, M)
    return dist


class ShapeCorr:
    def __init__(self, hparams, **kwargs):
        load_hparams = vars(hparams) if isinstance(hparams, Namespace) else hparams
        self.hparams = load_hparams

        self.accuracy = AccuracyAssumeEye()
        self.tracks = []

    def log_test_step(self, track_dict):
        logs_step = {k: to_numpy(v) for k, v in track_dict.items()}
        self.tracks.append(logs_step)

    def dataloader(self, dataset):
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False,
        )
        return loader

    @staticmethod
    def extract_labels_for_test(test_batch):
        data_dict = test_batch
        if "label_flat" in test_batch:
            label = data_dict["label_flat"].squeeze(0)
            pinput1 = data_dict["src_flat"]
            input2 = data_dict["tgt_flat"]
        else:
            pinput1 = data_dict["source"]["pos"]
            input2 = data_dict["target"]["pos"]
            label = matrix_map_from_corr_map(
                data_dict["gt_map"].squeeze(0), pinput1.squeeze(0), input2.squeeze(0)
            )

        ratio_list, soft_labels = PointCloudDataset.extract_soft_labels_per_pair(
            label, input2.squeeze(0), replace_on_cpu=True
        )

        return label, pinput1, input2, ratio_list, soft_labels

    def compute_acc(self, label, ratio_list, soft_labels, p, input2, dataset_name, map_start_index):
        track_dict = {}
        corr_tensor = ShapeCorr._prob_to_corr_test(p)
        hit = label.argmax(-1).squeeze(0)
        pred_hit = p.squeeze(0).argmax(-1)
        target_dist = square_distance(input2.squeeze(0), input2.squeeze(0))
        track_dict["acc_mean_dist"] = target_dist[pred_hit, hit].mean().item()
        if dataset_name == "tosca":
            track_dict[
                "acc_mean_dist"
            ] /= 3  # TOSCA is not scaled to meters as the other datasets. /3 scales the shapes to be coherent with SMAL (animals as well)
        if dataset_name == "shrec20":
            corr_tensor = corr_tensor[:,:map_start_index,:]
            label = label[:map_start_index,:]
        acc_000 = ShapeCorr._label_ACC_percentage_for_inference(
            corr_tensor, label.unsqueeze(0)
        )
        track_dict["acc_0.00"] = acc_000.item()
        for idx, ratio in enumerate(ratio_list):
            soft_label_ratio = soft_labels[f"{ratio}"].unsqueeze(0)
            if dataset_name == "shrec20":
                soft_label_ratio = soft_label_ratio[:,:map_start_index,:]
            track_dict[
                "acc_" + str(ratio)
            ] = ShapeCorr._label_ACC_percentage_for_inference(
                corr_tensor, soft_label_ratio
            ).item()
        return track_dict

    @staticmethod
    def _label_ACC_percentage_for_inference(label_in, label_gt):
        assert label_in.shape == label_gt.shape
        bsize = label_in.shape[0]
        label_in = label_in.cuda()
        b_acc = []
        for i in range(bsize):
            element_product = torch.mul(label_in[i], label_gt[i].cuda())
            N1 = label_in[i].shape[0]
            sum_row = torch.sum(element_product, dim=-1)  # N1x1

            hit = (sum_row != 0).sum()
            acc = hit.float() / torch.tensor(N1).float()
            b_acc.append(acc * 100.0)
        mean = torch.mean(torch.stack(b_acc))
        return mean

    @staticmethod
    def _prob_to_corr_test(prob_matrix):
        c = torch.zeros_like(prob_matrix)
        idx = torch.argmax(prob_matrix, dim=2, keepdim=True)
        for bsize in range(c.shape[0]):
            for each_row in range(c.shape[1]):
                c[bsize][each_row][idx[bsize][each_row]] = 1.0

        return c

    def test_step(self, batch, similarity, dataset_name):
        (
            label,
            pinput1,
            input2,
            ratio_list,
            soft_labels,
        ) = self.extract_labels_for_test(batch)
        track_dict = self.compute_acc(
            label, ratio_list, soft_labels, similarity, input2, dataset_name, batch["target"].get("map_start_index")
        )
        self.log_test_step(track_dict)
