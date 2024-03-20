import numpy as np
import torch
from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping


def compute_surface_map(path_1, path_2, c1, c2, source_index=None, target_index=None, use_wks=False, device=torch.device("cuda:0")):
    mesh1 = TriMesh(path_1)
    mesh2 = TriMesh(path_2)
    print("mesh1", mesh1.vertlist.shape)
    print("mesh2", mesh2.vertlist.shape)
    if not use_wks:
        process_params = {
        'n_ev': (50,50),  # Number of eigenvalues on source and Target
        'n_descr': 2048,
        'landmarks': None,
        'descr1': c1,
        'descr2': c2,
        'subsample_step': 0
        }
    else:
        process_params = {
        'n_ev': (50,50),  # Number of eigenvalues on source and Target
        'n_descr': 2048,
        'landmarks': None,
        'subsample_step': 1,  # In order not to use too many descriptors
        'descr_type': 'WKS',  # WKS or HKS
        'subsample_step': 0
        }
    model = FunctionalMapping(mesh1, mesh2)
    model.preprocess(**process_params,verbose=True)
    fit_params = {
    'w_descr': 1e0,
    'w_lap': 1e-2,
    'w_dcomm': 1e-1,
    'w_orient': 0
    }
    model.fit(**fit_params, verbose=True)
    p = model.get_p2p(n_jobs=1)
    if source_index is not None:
        p = p[source_index]
    p = torch.from_numpy(mesh1.vertices[p]).to(device)
    if target_index is not None:
        vertices = torch.from_numpy(mesh1.vertices[target_index]).to(device)
        p = torch.cdist(p, vertices)
        p = torch.argmin(p, dim=2)[0]
    else:
        vertices = torch.from_numpy(mesh1.vertices).to(device)
        p = torch.cdist(p, vertices)
        p = torch.argmin(p, dim=1)
    return p
