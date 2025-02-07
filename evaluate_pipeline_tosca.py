import torch
import glob
from tqdm import tqdm
from setup_args import default_arg_parser, init_parse_argparse_default_params
from dataloaders.tosca import TOSCA
from dataloaders.point_cloud_dataset import PointCloudDataset
# from visualization import visualize_pair_corr
from correspondence import ShapeCorr
import torch
import glob
from utils import cosine_similarity, solve_correspondence, gmm
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path 
import os


results_path = "output/TOSCA"
num_points = 1024
save_path = "results/" + results_path.split("/")[-1] + "_num_points_" + str(num_points)
save_path = save_path+".csv"
print("Saving in ", save_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = sorted([str(path) for path in list(Path(results_path).rglob("*.pt"))],key=lambda p: int(os.path.basename(p)[:-3]),)


# Make sure to keep the dataset at data/datasets/{dataset_name} or modify inside PointCloudDataset

def main(): 
    parser = default_arg_parser(description="Point correspondence")
    dataset_name = "tosca"
    task_name = "shape_corr"
    init_parse_argparse_default_params(parser, dataset_name)
    parser = PointCloudDataset.add_dataset_specific_args(
        parser, task_name, dataset_name, is_lowest_leaf=False
    )
    args = parser.parse_args()
    dataset = TOSCA(args, "test")
    shape_correspondence = ShapeCorr(args)
    dataloader = shape_correspondence.dataloader(dataset)
    pairs = []
    for data in tqdm(dataloader):
        for key, val in data["source"].items():
            data["source"][key] = val.cuda()
        for key, val in data["target"].items():
            data["target"][key] = val.cuda()
        data["gt_map"] = data["gt_map"].cuda()
        path_source = paths[data['source']['id']]
        path_target = paths[data['target']['id']]

        f_source = torch.load(path_source, map_location="cpu")
        f_target = torch.load(path_target, map_location="cpu")
        source_index = data["source"]["rand_choice"].int().cpu()
        target_index = data["target"]["org"].int().cpu()

        f_source = f_source[source_index].squeeze()
        f_target = f_target[target_index].squeeze()
        p = cosine_similarity(f_source.cuda(), f_target.cuda())

        pairs.append(f'{data["source"]["id"].item()}_{data["target"]["id"].item()}')
        shape_correspondence.test_step(data, p.unsqueeze(0), args.dataset_name)


    df = pd.DataFrame(shape_correspondence.tracks)
    df.insert(0,"pair",pairs)
    df.to_csv(save_path, index=False)

    print("see stats at generated csv, compute mean accuracy and error from there")


if __name__ == "__main__":
    main()
