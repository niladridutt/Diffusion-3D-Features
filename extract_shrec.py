import torch
import os
import glob
from diff3f import get_features_per_vertex
from time import time
from utils import convert_mesh_container_to_torch_mesh
import configparser
from datetime import datetime
from dataloaders.mesh_container import MeshContainer
from diffusion import init_pipe
from dino import init_dino


config = configparser.ConfigParser()
now = datetime.now()
date_time = now.strftime("%d-%m-%Y_%H-%M")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
min_points = 10000
num_samples = "all"
num_views = 100
H = 512
W = 512

make_prompt_null = False
num_images_per_prompt = 1
ensemble_size = 1
divide_by_sum = False
tolerance = 0.004
random_seed = 42
rotated = False
use_normal_map = True
dataset = "SHREC"
prompt = "human"
target_path = (
    "datasets/shrec/off_2/*.off"
)
save_path = (
    f"output/shrec/{date_time}"
)
add_path = dataset

if rotated:
    add_path += "-rotated"
save_path += add_path
if not os.path.isdir(save_path):
    os.makedirs(save_path)
target_files = sorted(glob.glob(target_path))


def write_config_file():
    config["Parameters"] = {
        "dataset": dataset,
        "method": "dino",
        "rotated": str(rotated),
        "prompt": prompt,
        "num_samples": str(num_samples),
        "num_views": str(num_views),
        "H": str(H),
        "W": str(W),
        "tolerance": str(tolerance),
        "make_prompt_null": str(make_prompt_null),
        "num_images_per_prompt": str(num_images_per_prompt),
        "ensemble_size": str(ensemble_size),
        "divide_by_sum": str(divide_by_sum),
        "target_path": target_path,
        "save_path": save_path,
        "random_seed": str(random_seed),
        "min_points": str(min_points),
        "use_normal_map": str(use_normal_map),
    }
    print(config)
    # Write the config file to disk
    with open(f"configs/config{add_path}-{date_time}.ini", "w") as f:
        config.write(f)
    with open(f"{save_path}/config{add_path}-{date_time}.ini", "w") as f:
        config.write(f)


def compute_features():
    t1 = time()
    pipe = init_pipe(device)
    dino_model = init_dino(device)
    for file in target_files:
        try:
            print(f"Processing {file}")
            filename = file.split("/")[-1].split(".")[0]
            m = MeshContainer().load_from_file(str(file))
            mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
            mesh_vertices = mesh.verts_list()[0]
            features = get_features_per_vertex(
                device=device,
                pipe=pipe, 
                dino_model=dino_model,
                mesh=mesh,
                prompt=prompt,
                mesh_vertices=mesh_vertices,
                num_views=num_views,
                H=H,
                W=W,
                tolerance=tolerance,
                num_images_per_prompt=num_images_per_prompt,
                use_normal_map=use_normal_map,
            )
            save_name = f"{save_path}/{filename}.pt"
            torch.save(features, save_name)
        except Exception as e:
            print(e)
    time_taken = (time() - t1) / 60
    print("Time taken to complete in minutes", time_taken)


if __name__ == "__main__":
    write_config_file()
    compute_features()
