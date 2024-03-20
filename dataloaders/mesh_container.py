import numpy as np
import scipy.io as sio
import os
import pickle
from plyfile import PlyData
import trimesh


class MeshContainer(object):
    """
    Helper class to store face, vert as numpy

    Control I/O, convertions and other mesh utilities
    """
    def __init__(self, vert=None, face=None):
        if (vert is None and face is None): return
        if (not type(vert).__module__ == np.__name__):
            vert = vert.cpu().detach().numpy()
        if (face is not None and not type(face).__module__ == np.__name__):
            face = face.cpu().detach().numpy().astype(np.int32)
        if (face is not None and face.shape[0] == 3): face = face.transpose()

        self.vert, self.face = vert, face
        self.n = self.vert.shape[0]

    def copy(self):
        return MeshContainer(self.vert.copy(), self.face.copy())

    def _load_from_file_mat(self, file_path, dataset='faust_reg'):
        self.load_file_name = file_path
        mesh_mat = sio.loadmat(file_path)

        if (dataset == 'tosca'):
            vx = mesh_mat['surface']['X'][0, 0]
            vy = mesh_mat['surface']['Y'][0, 0]
            vz = mesh_mat['surface']['Z'][0, 0]
            face = np.array(mesh_mat['surface']['TRIV'][0, 0].astype(np.int32))
        elif ('null' in file_path
              or ('partial' in file_path and 'rescaled' not in file_path)):
            points = mesh_mat['N']['xyz'][0][0]
            vx = points[:, 0][:, None]
            vy = points[:, 1][:, None]
            vz = points[:, 2][:, None]
            face = np.array(mesh_mat['N']['tri'][0][0].astype(np.int32))
        elif ('model' in file_path and 'remesh' in file_path):
            vx = mesh_mat['part']['X'][0, 0]
            vy = mesh_mat['part']['Y'][0, 0]
            vz = mesh_mat['part']['Z'][0, 0]
            face = np.array(mesh_mat['part']['triv'][0, 0].astype(np.int32))
        elif ('remesh' in file_path and 'deform' not in file_path):
            vx = mesh_mat['model_remesh']['X'][0, 0]
            vy = mesh_mat['model_remesh']['Y'][0, 0]
            vz = mesh_mat['model_remesh']['Z'][0, 0]
            face = np.array(mesh_mat['model_remesh']['triv'][0, 0].astype(
                np.int32))
        else :
            vx = mesh_mat['VERT'][:, 0][:, None]
            vy = mesh_mat['VERT'][:, 1][:, None]
            vz = mesh_mat['VERT'][:, 2][:, None]
            face = np.array(mesh_mat['TRIV'].astype(np.int32))

        if np.min([face]) > 0:
            face = face - 1
        self.face = face
        self.vert = np.concatenate((vx, vy, vz), axis=1)
        self.n = self.vert.shape[0]

    def _load_from_file_ply(self, file_path):
        self.load_file_name = file_path
        ply_data = PlyData.read(file_path)
        vx = np.array(ply_data['vertex'].data['x'])[:, np.newaxis]
        vy = np.array(ply_data['vertex'].data['y'])[:, np.newaxis]
        vz = np.array(ply_data['vertex'].data['z'])[:, np.newaxis]
        self.vert = np.concatenate((vx, vy, vz), axis=1)
        self.face = ply_data['face'].data['vertex_indices'][:]
        if (len(self.face.shape) < 2):
            self.face = np.array([x for x in self.face])
        self.n = self.vert.shape[0]

    def _load_from_file_obj(self, file_path):
        try:
            with open(file_path, 'r') as f:
                vertices = []
                faces = []
                for line in f:
                    line = line.strip()
                    if line == '' or line[0] == '#':
                        continue
                    line = line.split()
                    if line[0] == 'v':
                        vertices.append([float(x) for x in line[1:]])
                    elif line[0] == 'f':
                        faces.append([int(x.split('/')[0]) - 1 for x in line[1:]])
            self.vert, self.face = np.asarray(vertices), np.asarray(faces)
            self.n = self.vert.shape[0]
        except:
            import trimesh
            mesh = trimesh.exchange.obj.load_obj(open(file_path,'rb'))
            self.vert = mesh['vertices']
            self.face = mesh['faces']
            self.n = self.vert.shape[0]
            return


        # import trimesh
        # mesh = trimesh.exchange.obj.load_obj(open(file_path,'rb'))
        # self.vert = mesh['vertices']
        # self.face = mesh['faces']
        # self.n = self.vert.shape[0]
        # return
        # file = open(file_path, 'r+')
        # os.system('meshlabserver -i ' + file_path + ' -o ' +
        #           os.path.basename(file_path)[:-4] + '.ply')
        # return self._load_from_file_ply(
        #     os.path.basename(file_path)[:-4] + '.ply')
    
    def _load_from_file_off(self, file_path):
        import trimesh
        mesh = trimesh.exchange.off.load_off(open(file_path,'rb'))
        self.vert = mesh['vertices']
        self.face = mesh['faces']
        self.n = self.vert.shape[0]


    def _load_from_raw(self, file_path):
        self.load_file_name = file_path
        shape = pickle.load(open(file_path, 'rb'))
        container = MeshContainer(shape['pos'], shape['face'])

        try:
            container.dist = shape['geodesic_dist_map']
        except:
            pass
        return container

    def load_from_file(self, file_path, dataset=''):
        self.file_path = file_path
        filename = os.path.basename(file_path)
        if filename.endswith('mat'):
            self._load_from_file_mat(file_path, dataset)
        elif filename.endswith('ply'):
            self._load_from_file_ply(file_path)
        elif filename.endswith('obj'):
            self._load_from_file_obj(file_path)
        elif filename.endswith('off'):
            self._load_from_file_off(file_path)
        elif (filename.isnumeric()):
            self = self._load_from_raw(file_path)
        return self

    def save_to_ply_and_obj(self):
        vertex = np.array([tuple(i) for i in self.vert],
                          dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        face = np.array(
            [(tuple(i), ) for i in self.face],
            dtype=[("vertex_indices", "i4", (3, ))],
        )
        el = PlyElement.describe(vertex, "vertex")
        el2 = PlyElement.describe(face, "face")
        plydata = PlyData([el, el2])
        plydata.write(self.load_file_name[:-4] + '.ply')

        import trimesh
        mesh1 = trimesh.load(self.load_file_name[:-4] + '.ply')
        with open(self.load_file_name[:-4] + '.obj', "w") as text_file:
            text_file.write(trimesh.exchange.obj.export_obj(mesh1))

    def save_as_mat(self, file_path=''):
        face_to_save = self.face if (np.min(self.face) > 0) else (self.face +
                                                                  1)
        mesh = {'face': face_to_save.tolist(), 'VERT': self.vert.tolist()}
        if (file_path == ''):
            file_path = self.file_path
        sio.savemat(file_path[:-4] + '.mat', mesh)

    def get_area_of_faces(self):
        """
        Compute the areas of all triangles on the mesh.
        Parameters
        ----------
        Returns
        -------
        area: 1-D numpy array
            area[i] is the area of the i-th triangle
        """
        areas = np.zeros(self.face.shape[0])

        for i, triangle in enumerate(self.face):
            a = np.linalg.norm(self.vert[triangle[0]] - self.vert[triangle[1]])
            b = np.linalg.norm(self.vert[triangle[1]] - self.vert[triangle[2]])
            c = np.linalg.norm(self.vert[triangle[2]] - self.vert[triangle[0]])
            s = (a + b + c) / 2.0
            areas[i] = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return areas

    def compute_LBO(self, num_evecs=15):
        spectrum_results = fem_laplacian(self.vert,
                                         self.face,
                                         num_evecs,
                                         normalization="areaindex",
                                         areas=self.get_area_of_faces())
        self_functions = spectrum_results['self_functions'].copy()
        return self_functions