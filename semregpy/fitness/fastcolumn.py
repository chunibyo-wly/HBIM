import numpy as np
import open3d as o3d

from core.fitness.interface import *

from gprMax.tools.outputfiles_merge import get_output_data


class PCD:
    origin = None
    display = None
    patch_horizontal = None
    kdtree_horizontal = None

    def __init__(self, pcd, display_vox=0.02):
        self.origin = pcd
        # normal
        if not self.origin.has_normals():
            self.origin.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
        print("origin: ", self.origin)
        # display
        self.display = self.origin.voxel_down_sample(voxel_size=display_vox)
        print("display: ", self.display)
        # patch
        h = []
        for i in range(len(self.origin.points)):
            if (
                self.origin.normals[i][2] < 0.5
                and self.origin.normals[i][2] > -0.5
            ):
                h.append(i)
        self.patch_horizontal = self.origin.select_by_index(h)
        print("patch_horizontal: ", self.patch_horizontal)
        # tree
        self.kdtree_horizontal = o3d.geometry.KDTreeFlann(self.patch_horizontal)

    def search(self, p):
        return self.kdtree_horizontal.search_knn_vector_3d(
            p, 1
        )  # [k, idx, dist]


class Mesh:
    pcd = None
    mesh = None
    center = None

    def __init__(self, mesh, sample_vox=0.01, display_vox=0.02):
        self.mesh = mesh
        # sample
        max = mesh.get_max_bound()
        min = mesh.get_min_bound()
        dx = max[0] - min[0]
        dy = max[1] - min[1]
        dz = max[2] - min[2]
        num = int((dx * dy + dy * dz + dz * dx) * 2 / sample_vox**2)
        self.pcd = PCD(mesh.sample_points_uniformly(number_of_points=num))


class FastColumn(FitnessInterface):
    prob = None
    comp = None
    obj_size = 0

    def __init__(self, prob=None, toMaximize=False):
        super().__init__(toMaximize)
        self.obj = None
        if prob is not None:
            self.set_prob(prob)

    def set_prob(self, pcd):
        self.prob = PCD(pcd)

    def read_prob(self, fname: str):
        self.set_prob(o3d.io.read_point_cloud(fname))

    def set_comp(self, fname: str):
        mesh = o3d.io.read_triangle_mesh(fname, enable_post_processing=True)
        mesh.scale(0.001, [0, 0, 0])
        self.comp = Mesh(mesh)
