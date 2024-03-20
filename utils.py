import torch
import numbers
import numpy as np
import argparse
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from PIL import Image
from pytorch3d.io import load_obj
import matplotlib.pyplot as plt
import meshplot as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys


def generate_colors(n):
    hues = [i / n for i in range(n)]
    saturation = 1
    value = 1
    colors = [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors


def plot_mesh(myMesh,cmap=None):
    mp.plot(myMesh.vert, myMesh.face,c=cmap)
    
def double_plot(myMesh1,myMesh2,cmap1=None,cmap2=None):
    d = mp.subplot(myMesh1.vert, myMesh1.face, c=cmap1, s=[2, 2, 0])
    mp.subplot(myMesh2.vert, myMesh2.face, c=cmap2, s=[2, 2, 1], data=d)

def get_colors(vertices):
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return cmap


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy()"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError


def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError


def convert_trimesh_to_torch_mesh(tm, device, is_tosca=True):
    verts_1, faces_1 = torch.tensor(tm.vertices, dtype=torch.float32), torch.tensor(
        tm.faces, dtype=torch.float32
    )
    if is_tosca:
        verts_1 = verts_1 / 10
    verts_rgb = torch.ones_like(verts_1)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts_1], faces=[faces_1], textures=textures)
    mesh = mesh.to(device)
    return mesh

def convert_mesh_container_to_torch_mesh(tm, device, is_tosca=True):
    verts_1, faces_1 = torch.tensor(tm.vert, dtype=torch.float32), torch.tensor(
        tm.face, dtype=torch.float32
    )
    if is_tosca:
        verts_1 = verts_1 / 10
    verts_rgb = torch.ones_like(verts_1)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts_1], faces=[faces_1], textures=textures)
    mesh = mesh.to(device)
    return mesh

def load_textured_mesh(mesh_path, device):
    verts, faces, aux = load_obj(mesh_path)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # Initialise the mesh with textures
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=tex)
    mesh = mesh.to(device)
    return mesh

def cosine_similarity(a, b):
    if len(a) > 30000:
        return cosine_similarity_batch(a, b, batch_size=30000)
    dot_product = torch.mm(a, b.t())
    norm_a = torch.norm(a, dim=1, keepdim=True)
    norm_b = torch.norm(b, dim=1, keepdim=True)
    similarity = dot_product / (norm_a * norm_b.t())

    return similarity


def cosine_similarity_batch(a, b, batch_size=30000):
    num_a, dim_a = a.size()
    num_b, dim_b = b.size()
    similarity_matrix = torch.empty(num_a, num_b, device="cpu")
    for i in tqdm(range(0, num_a, batch_size)):
        a_batch = a[i:i+batch_size]
        for j in range(0, num_b, batch_size):
            b_batch = b[j:j+batch_size]
            dot_product = torch.mm(a_batch, b_batch.t())
            norm_a = torch.norm(a_batch, dim=1, keepdim=True)
            norm_b = torch.norm(b_batch, dim=1, keepdim=True)
            similarity_batch = dot_product / (norm_a * norm_b.t())
            similarity_matrix[i:i+batch_size, j:j+batch_size] = similarity_batch.cpu()
    return similarity_matrix


def hungarian_correspondence(similarity_matrix):
    # Convert similarity matrix to a cost matrix by negating the similarity values
    cost_matrix = -similarity_matrix.cpu().numpy()

    # Use the Hungarian algorithm to find the best assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Create a binary matrix with 1s at matched indices and 0s elsewhere
    num_rows, num_cols = similarity_matrix.shape
    match_matrix = np.zeros((num_rows, num_cols), dtype=int)
    match_matrix[row_indices, col_indices] = 1
    match_matrix = torch.from_numpy(match_matrix).cuda()
    return match_matrix


def gmm(a, b):
    # Compute Gram matrices
    gram_matrix_a = torch.mm(a, a.t())
    gram_matrix_b = torch.mm(b, b.t())

    # Expand dimensions to facilitate broadcasting
    gram_matrix_a = gram_matrix_a.unsqueeze(1)
    gram_matrix_b = gram_matrix_b.unsqueeze(0)

    # Compute Frobenius norm for each pair of vertices using vectorized operations
    correspondence_matrix = torch.norm(gram_matrix_a - gram_matrix_b, p='fro', dim=2)

    return correspondence_matrix
