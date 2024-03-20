from data.point_cloud_db.point_cloud_dataset import PointCloudDataset, get_max_dist
import itertools
import os
import os.path
import numpy as np

from tqdm.auto import tqdm
from visualization.mesh_container import MeshContainer
import math



def download(faust_data_path):
    os.makedirs(faust_data_path,exist_ok=True)
    zip_path = os.path.join(faust_data_path, 'MPI-FAUST.zip')
    if not os.path.exists(zip_path[:-4]):
        download_command = f"wget --header=\"Host: faust.is.tue.mpg.de\" --header=\"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36\" --header=\"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9\" --header=\"Accept-Language: en-US,en;q=0.9,he-IL;q=0.8,he;q=0.7\" --header=\"Referer: http://faust.is.tue.mpg.de/challenge/Inter-subject_challenge/datasets\" --header=\"Cookie: auth_token=43KA9BlRS7pvMtNOogtFNQ; _faust_session=BAh7B0kiD3Nlc3Npb25faWQGOgZFRkkiJTIyYTcxMjExYTEwM2VhNjFjOWM3ZWMyM2I0Njk5OTVjBjsAVEkiEF9jc3JmX3Rva2VuBjsARkkiMWpXOEJsRDVxRnpSck02WnJpU0hDSHMzRVNtLy9Pa2hpV0ZwVmpqeTgvaW89BjsARg%3D%3D--18e891f93f3852775a5cd42b7aabd497f29298ec\" --header=\"Connection: keep-alive\" \"http://faust.is.tue.mpg.de/main/download\" -c -O {zip_path}"
        os.system(f'{download_command}')
        os.system(f'unzip {zip_path} -d {faust_data_path}')
        os.system(f'rm {zip_path}')




class FAUST(PointCloudDataset):
    def __init__(self, params, split='train'):
        super(FAUST, self).__init__(params,split)

    @staticmethod
    def add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = PointCloudDataset.add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False)
        return parser
    
    
    def valid_pairs(self,gt_map):
        num_shapes = int(math.sqrt(gt_map.shape[0]))
        all_pairs = list(itertools.product(list(range(num_shapes)), list(range(num_shapes))))
        return list(filter(lambda pair: pair[0] != pair[1],all_pairs))

    @staticmethod
    def load_data(data_root, split,hparams):
        if(not os.path.exists(f"{data_root}/MPI-FAUST")):
            download(data_root)
        data_dir = os.path.join(data_root, "MPI-FAUST", "training", "registrations")

        all_data_file_name = "all_samples.npy"
        all_data_path = os.path.join(data_dir, all_data_file_name)
        if not os.path.exists(all_data_path):
            file_base_name = "tr_reg_%03d.ply"
            mesh = None
            all_verts = []
            all_d_max = []
            for file_idx in tqdm(range(100)):
                file_path = os.path.join(data_dir, file_base_name % file_idx)
                mesh = MeshContainer().load_from_file(file_path)
                all_verts.append(mesh.vert)

                d_max = get_max_dist(mesh.vert)
                all_d_max.append(d_max)

            np.save(all_data_path, (mesh.face, all_verts, all_d_max))
            face = mesh.face
        else:
            face, all_verts, all_d_max = np.load(all_data_path,allow_pickle=True)

        if split == "train":
            all_verts = all_verts[:80]
            all_d_max = all_d_max[:80]
        else:
            all_verts = all_verts[80:]
            all_d_max = all_d_max[80:]

        all_verts = np.array(all_verts)
        faces_repeated = np.broadcast_to(face[None,:], (all_verts.shape[0], *face.shape))

        # The gt map for faust is all for all
        gt_is_eye =  np.broadcast_to(np.arange(all_verts.shape[1])[None,:], (all_verts.shape[0] ** 2, all_verts.shape[1]))
        return all_verts,faces_repeated,all_d_max, gt_is_eye

if __name__ == "__main__":
    pass
