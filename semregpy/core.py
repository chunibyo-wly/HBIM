import copy
import os
import random
import time
import tempfile

import nlopt
import numpy as np
import open3d as o3d

from semregpy import (
    ColumnComponent,
    DoorComponent,
    OfficeComponent,
    DougongComponent,
)
from utils import Settings, settings

temp_dir = tempfile.gettempdir()
log_prefix = os.path.join(temp_dir, "cma_temp_")  # Prefix for log files


def random_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return [r, g, b]


def npxyz_to_pcd(np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np)
    return pcd


def nth_neibour_distance(points, nth=4):
    """
    return: mean_4th_distance, std_distance, eps
    """

    pcd = o3d.geometry.PointCloud()
    points = points[:, :3]
    pcd.points = o3d.utility.Vector3dVector(points)

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    total_distance = 0
    num_distances = 0
    distances_list = []

    for point in points:
        _, indices, distances = kdtree.search_knn_vector_3d(point, nth)

        nearest_neighbor_index = indices[-1]
        nearest_neighbor_distance = np.sqrt(distances[-1])

        distances_list.append(nearest_neighbor_distance)
        total_distance += nearest_neighbor_distance
        num_distances += 1

    mean_4th_distance = np.mean(distances_list)
    std_distance = np.std(distances_list)
    eps = mean_4th_distance + 2 * std_distance
    return mean_4th_distance, std_distance, eps


class PCD:

    origin = None  # 原始点云
    patch = None  # 水平和垂直点云， 0 = horizontal, 1 = vertical
    kdtree = None  # kdtree， 0 = horizontal, 1 = vertical
    WEIGHT_NORNAL_ERROR = 0.2  # normal error 权重

    def __init__(self, pcd, control_size=None, cluster=False, split_patch=True):
        self.control_size = control_size
        self.origin = pcd  # .voxel_down_sample(voxel_size=0.02)

        if cluster:
            # self.z_ratio, 移除上下z比例
            self.z_ratio = 0.4
            # 移除上下点云cluster列表
            patch_remove_points = self.remove_outliers(
                self.origin, z_ratio=self.z_ratio
            )
            # 切除上下点云后-dbscan聚类完成再补充切除点云， 恢复点云中心列表
            self.patch_recover, self.cluster_center_list = (
                self.recover_outliers(
                    patch_remove_points, self.origin, z_ratio=self.z_ratio
                )
            )
            if Settings.GUI:
                for i, p in enumerate(self.patch_recover):
                    p.paint_uniform_color(random_color())
                    o3d.io.write_point_cloud(f"cluster{i}.ply", p)
            print("pointcloud has ", len(self.patch_recover), "cluster")

        if split_patch:
            # 对原始点云进行h,v拆分
            self.split_patch()

    def split_patch(self):
        # normal
        if not self.origin.has_normals():
            # print('normal estimation')
            self.origin.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.2, max_nn=30
                )
            )
        h = []
        v = []
        for i in range(len(self.origin.points)):
            nz = self.origin.normals[i][2]
            if nz < 0.5 and nz > -0.5:
                h.append(i)
            elif nz > 0.5 or nz < -0.5:
                v.append(i)
        # patch [0] = horizontal, [1] = vertical points
        # self.origin
        self.patch = [None, None]
        self.kdtree = [None, None]

        # self.kdtree[0], self.kdtree[1] = self.get_points_patch(self.origin)

        self.kdtree[0] = self.origin.select_by_index(h)
        self.kdtree[1] = self.origin.select_by_index(v)

        # control size for components
        if self.control_size is not None:
            for i in [0, 1]:
                vox = 1
                while (
                    self.patch[i] is None
                    or len(self.patch[i].points) <= self.control_size
                ):
                    vox /= 1.3
                    self.patch[i] = self.kdtree[i].voxel_down_sample(
                        voxel_size=vox
                    )
        else:
            self.patch[0] = self.kdtree[0]
            self.patch[1] = self.kdtree[1]

        self.kdtree[0] = o3d.geometry.KDTreeFlann(self.patch[0])
        self.kdtree[1] = o3d.geometry.KDTreeFlann(self.patch[1])

    def resort_clusters(self, clusters, target_points):

        v = target_points.get_axis_aligned_bounding_box().get_extent()
        ratio_v = v[2] / v[0]

        new_pcd = o3d.geometry.PointCloud()
        new_cluster_list = []  # 恢复点云列表
        new_pcd_list_inds = []  # 恢复点云列表索引
        new_center_list = []  # 恢复点云中心列表
        for i, cluster in enumerate(clusters):
            v_cluster = cluster.get_axis_aligned_bounding_box().get_extent()

            ratio_cluster = v_cluster[2] / v_cluster[0]

            if (
                ratio_cluster / ratio_v <= 1.3
                and ratio_cluster / ratio_v >= 0.7
            ):
                cluster.paint_uniform_color([0, 0, i / len(clusters)])
                cluster.paint_uniform_color([0, 0, 1])
                new_pcd += cluster
                new_cluster_list.append(cluster)
                new_center_list.append(
                    np.mean(np.array(cluster.points), axis=0)
                )
            else:
                new_pcd_list_inds.append(i)

        for i in new_pcd_list_inds:
            clusters[i].paint_uniform_color([1, 0, 0])

            new_pcd += clusters[i]
            new_center_list.append(
                np.mean(np.array(clusters[i].points), axis=0)
            )
            new_cluster_list.append(clusters[i])

        self.origin = new_pcd
        self.patch_recover = new_cluster_list
        self.cluster_center_list = np.array(new_center_list)

        # for i in self.cluster_center_list:
        #   pcd = npxyz_to_pcd(i.reshape(1,3))
        #   pcd.paint_uniform_color([0,1,0])
        #   new_pcd += pcd

        # o3d.visualization.draw_geometries([new_pcd])

    def split_sort(self, data, label=6):
        """
        根据label对数据进行排序
        """
        sorted_indices = np.argsort(data[:, label])
        points = data[sorted_indices]
        indices = np.unique(points[:, label], return_index=True)[1]
        indices = np.sort(indices)[1:]
        return np.split(points, indices)

    def recover_outliers(self, points, ori_points, z_ratio=0.8):
        """
        根据dbscan cluster的结果，将cluster中z轴上的点云恢复成原始点云
        输入：points：cluster后的点云，ori_points：原始点云，z_ratio：移除上下点云的比例
        输出：point_list：恢复后的点云列表，cluster_center_list：恢复后的点云cluster中心列表
        """
        cluster_new = self.cluster(points)
        cluster_new = self.split_sort(cluster_new, label=-1)

        point_list = []
        kdtree_list = []
        cluster_center_list = []

        for i in cluster_new:
            if i.shape[0] > min(len(points.points) * 0.001, 1000):
                i_pcd = npxyz_to_pcd(i[:, :3])
                i_box = i_pcd.get_axis_aligned_bounding_box()
                i_box_extend = i_box.get_extent()
                i_box_extend[0] = i_box_extend[0] * 1.05
                i_box_extend[1] = i_box_extend[1] * 1.05
                i_box_extend[2] = i_box_extend[2] / z_ratio

                i_box_center = i_box.get_center()
                box = o3d.geometry.AxisAlignedBoundingBox(
                    i_box_center - i_box_extend / 2,
                    i_box_center + i_box_extend / 2,
                )
                i_new_pcd = ori_points.crop(box)
                i_new_pcd.paint_uniform_color([1, 0, 0])

                i_new_kdtree = o3d.geometry.KDTreeFlann(i_new_pcd)

                kdtree_list.append(i_new_kdtree)
                point_list.append(i_new_pcd)
                cluster_center_list.append(
                    np.mean(np.array(i_new_pcd.points), axis=0)
                )
        point_list = point_list[1:]
        cluster_center_list = cluster_center_list[1:]
        kdtree_list = kdtree_list[1:]

        return point_list, np.array(cluster_center_list)

    def remove_outliers(self, points, z_ratio=0.8):
        """
        按z ratio移除上下点云
        """
        box = points.get_axis_aligned_bounding_box()
        box_extent = box.get_extent()
        box_center = box.get_center()

        new_box_extent = box_extent
        new_box_extent[2] = new_box_extent[2] * z_ratio
        box_new = o3d.geometry.AxisAlignedBoundingBox(
            box_center - new_box_extent / 2, box_center + new_box_extent / 2
        )
        points_new = points.crop(box_new)
        return points_new

    def cluster(self, pcd):
        xyz = np.array(pcd.points)
        eps = nth_neibour_distance(xyz)[2]
        labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=2, print_progress=False)
        ).reshape(-1, 1)
        # 获取噪声点的索引
        points = np.hstack((xyz, labels))
        return points

    def split(self, data):
        index_dict = {}  # 存储元素和索引的字典

        for i, num in enumerate(data[:, -1]):
            if num in index_dict:
                index_dict[num].append(i)
            else:
                index_dict[num] = [i]

        return index_dict

    def dist_normal(self, p, tid, n):
        [k, idx, r2] = self.kdtree[tid].search_knn_vector_3d(p, 1)
        if k == 1:
            cos_err = min(
                abs(1 - np.sum(n * self.patch[tid].normals[idx[0]])),
                abs(1 - np.sum((-n) * self.patch[tid].normals[idx[0]])),
            )
            dn = self.WEIGHT_NORNAL_ERROR * cos_err  # penalty from normal error
            return r2[0] + dn**2 + 2 * np.sqrt(r2[0]) * dn, idx[0]
        else:
            return 1e4  # not found

    def dist(self, p, tid):
        [k, idx, r2] = self.kdtree[tid].search_knn_vector_3d(p, 1)
        if k == 1:
            return r2[0]
        else:
            return 1e4  # not found


class Mesh:
    pcd = None
    mesh = None
    center = None

    def __init__(self, mesh):
        self.mesh = mesh
        self.mesh.translate(-self.mesh.get_center())

    def sample_pcd(self, w, depth=6):
        max = self.mesh.get_max_bound()
        min = self.mesh.get_min_bound()
        dx = max[0] - min[0]
        dy = max[1] - min[1]
        dz = max[2] - min[2]
        box_area = dx * dy + dy * dz + dz * dx
        interet_area = box_area
        if w[0] > 0:
            interet_area += dy * dz + dz * dx
        if w[1] > 0:
            if dy * dx < interet_area:
                interet_area += dy * dx
        target_num = (2**depth) ** 2.5
        num = int(target_num * box_area / interet_area)
        self.pcd = PCD(
            self.mesh.sample_points_uniformly(number_of_points=num),
            control_size=128,
            cluster=False,
        )


class SemRegPy:
    library = None
    algorithm = None
    # fit = None
    time_elapsed = 0.0
    VERBOSE = False
    window = None

    prob = None
    comp = None
    mesh = None

    results = []

    best_f = None
    best_x = None
    toMax = False
    iter = 0
    TABU_RANGE = 0.2

    def __init__(self, lib="nlopt"):
        self.library = lib

        self.pcd_cache_path = ""

    def solve(
        self,
        comp,
        max_eval=200,
        alg=Settings.opt_alg,
        random=False,
        resort=False,
    ):
        self.mesh.sample_pcd(comp.params["hv_weights"])
        self.iter = 0
        self.best_f = None
        self.best_x = None
        self.comp = comp
        self.comp.params["min"] = (
            self.prob.origin.get_min_bound()
            - self.mesh.pcd.origin.get_min_bound()
        )
        self.comp.params["max"] = (
            self.prob.origin.get_max_bound()
            - self.mesh.pcd.origin.get_max_bound()
        )
        self.comp.params["c_mesh"] = (
            self.mesh.mesh.get_max_bound() - self.mesh.mesh.get_min_bound()
        ) / 2
        self.algorithm = alg
        self.cluster_idx_0 = {}
        self.cluster_idx_1 = {}

        self.current_entity = None

        if resort:
            self.prob.resort_clusters(
                self.prob.patch_recover, self.mesh.pcd.origin
            )
            self.prob.split_patch()

        # ========= step 1 ============
        comp.set_stage(1)
        dim = comp.get_dof()
        if self.VERBOSE:
            print(
                "Problem: Horizontal points: ", len(self.prob.patch[0].points)
            )
            print("Problem: Vertical points: ", len(self.prob.patch[1].points))
            print(
                "Comp: Horizontal points: ", len(self.mesh.pcd.patch[0].points)
            )
            print("Comp: Vertical points: ", len(self.mesh.pcd.patch[1].points))
        if random:
            init_x = np.random.rand(dim)
        else:
            init_x = np.ones(dim) / 2
        if self.library == "nlopt":
            opt = nlopt.opt(alg, dim)
            opt.set_lower_bounds(np.zeros(dim))
            opt.set_upper_bounds(np.ones(dim))
            opt.set_min_objective(self.fitness)
            opt.set_maxeval(max_eval)
            opt.set_xtol_rel(3e-4)  # 0.1mm
            time_start = time.time()  # t_start
            x = opt.optimize(init_x)
            self.time_elapsed = time.time() - time_start  # t_end
            minf = opt.last_optimum_value()
        elif self.library == "cmaes":
            import cma

            # 设置初始参数
            lower_bounds = np.zeros(dim)
            upper_bounds = np.ones(dim)
            bounds = [lower_bounds, upper_bounds]
            sigma = 0.3  # 标准偏差，控制初始搜索范围
            opts = {
                "bounds": bounds,
                "maxfevals": max_eval,  # 最大评价次数
                "tolfunrel": 3e-4,  # 收敛容忍度（相对）
                "verb_filenameprefix": log_prefix,  # Write logs to temp folder
            }

            # 启动 CMA-ES 优化
            time_start = time.time()  # t_start
            result = cma.fmin(self.fitness, init_x, sigma, opts)  # 调用优化器
            self.time_elapsed = time.time() - time_start  # t_end

            # 获取结果
            x = result[0]  # 最优解
            minf = result[1]  # 最优目标函数值
        if self.VERBOSE:
            print(
                "step1 alg, max_eval, minf, self.time_elapsed:",
                alg,
                max_eval,
                minf,
                self.time_elapsed,
            )
            print("\t\t***:", self.best_f, self.best_x, dim)
        param = comp.decode_x(self.best_x)
        if self.VERBOSE:
            print(param)
        # if self.window is not None:
        #  shift = param['c'] - self.mesh.mesh.get_center()
        #  self.mesh.mesh.translate(shift)
        #  self.window.update_geometry(self.mesh.mesh)
        #  self.window.update_renderer()
        #  self.window.poll_events()
        # ========= step 2  ============
        comp.set_stage(2)
        self.best_f = 1e8
        dim = comp.get_dof()
        if random:
            init_x = np.random.rand(dim)
        else:
            init_x = np.ones(dim) / 2
        if self.library == "nlopt":
            opt = nlopt.opt(alg, dim)
            opt.set_lower_bounds(np.zeros(dim))
            opt.set_upper_bounds(np.ones(dim))
            opt.set_min_objective(self.fitness)
            opt.set_maxeval(int(max_eval / 2))
            opt.set_xtol_rel(3e-4)  # 0.1mm
            time_start = time.time()  # t_start
            x = opt.optimize(init_x)
            self.time_elapsed = time.time() - time_start  # t_end
            minf = opt.last_optimum_value()
        elif self.library == "cmaes":
            import cma

            # 定义边界和CMA-ES参数
            lower_bounds = np.zeros(dim)
            upper_bounds = np.ones(dim)
            bounds = [lower_bounds, upper_bounds]
            sigma = 0.3  # 初始标准偏差，控制搜索步长
            opts = {
                "bounds": bounds,  # 搜索范围：[0, 1]^dim
                "maxfevals": int(max_eval / 2),  # 设置最大评价次数
                "tolfunrel": 3e-4,  # 收敛相对容忍度
                "verb_filenameprefix": log_prefix,  # Write logs to temp folder
            }

            # 启动CMA-ES优化
            time_start = time.time()  # 记录开始时间
            result = cma.fmin(self.fitness, init_x, sigma, opts)  # 优化
            self.time_elapsed = time.time() - time_start  # 记录结束时间

            # 提取优化结果
            x = result[0]  # 最优解
            minf = result[1]  # 最优目标函数值
        if self.VERBOSE:
            print(
                "step2 alg, max_eval, minf, self.time_elapsed:",
                alg,
                max_eval,
                minf,
                self.time_elapsed,
            )
            print("\t\t***:", self.best_f, self.best_x)
        param = comp.decode_x(self.best_x)
        if self.VERBOSE:
            print("param:", param)
        param["best_f"] = self.best_f

        self.param = param

        # ! move z axis in hard code
        # TODO: should be removed in the future
        if comp.type == "Dougong":
            self.param["c"][2] = self.prob.patch[1].get_min_bound()[2]
        else:
            self.param["c"][2] = self.prob.origin.get_min_bound()[2]
        # ! move z axis in hard code

        return param

    def transformation(self, pcd, param):
        translation = param["best_c"]
        rotation = param["best_rz"]
        pcd_T = copy.deepcopy(pcd)
        R = pcd.get_rotation_matrix_from_xyz((0, 0, rotation))
        pcd_T.rotate(R, center=pcd.get_center())
        pcd_T.translate(translation)
        return pcd_T

    def update_kdtree(self, cnt=0):
        self.mesh_pcd = self.transformation(self.mesh.pcd.patch[0], self.param)
        if self.comp.type == "Column" or self.comp.type == "Dougong":
            obb_max_bound = self.mesh_pcd.get_max_bound()
            ratio = 0.8 if self.comp.type == "Column" else 2
            obb_max_bound[2] = (
                obb_max_bound[2] - self.mesh_pcd.get_min_bound()[2]
            ) * ratio + self.mesh_pcd.get_min_bound()[2]
            obb = o3d.geometry.AxisAlignedBoundingBox(
                self.mesh_pcd.get_min_bound(), obb_max_bound
            ).get_oriented_bounding_box()
        else:
            obb = self.mesh_pcd.get_oriented_bounding_box()
        obb_width = (obb.extent[0] + obb.extent[1]) / 2
        obb_tabu_width = obb_width + self.TABU_RANGE * 2
        obb.scale(obb_tabu_width / obb_width, center=obb.get_center())
        print("obb.extent: ", obb.extent)
        for hv_id in [0, 1]:
            self.prob.patch[hv_id] = self.prob.patch[hv_id].crop(
                obb, invert=True
            )
            self.prob.kdtree[hv_id] = o3d.geometry.KDTreeFlann(
                self.prob.patch[hv_id]
            )
        for i in range(len(self.prob.patch_recover)):
            self.prob.patch_recover[i] = self.prob.patch_recover[i].crop(
                obb, invert=True
            )

        if not Settings.GUI:
            o3d.io.write_point_cloud(
                f"{cnt}.ply", self.prob.patch[0] + self.prob.patch[1]
            )
        thre = 10 if self.comp.type == "Dougong" else 100
        print(
            "patch 0 ",
            len(self.prob.patch[0].points),
            "patch 1 ",
            len(self.prob.patch[1].points),
            "thre ",
            thre,
        )
        if (
            len(self.prob.patch[0].points) < thre
            and len(self.prob.patch[1].points) < thre
        ) or cnt >= len(self.prob.cluster_center_list):
            print("Points are not enough")
            return False
        else:
            return True

    def update_kdtree2(self, cnt=0):
        # final mesh position
        self.mesh_pcd = self.transformation(self.mesh.pcd.patch[0], self.param)
        mesh_pcd_center = self.mesh_pcd.get_center()
        # o3d.visualization.draw_geometries([self.mesh_pcd, self.prob.patch[0]])

        list_0 = []
        # match center of cluster to the center of mesh
        if self.prob.cluster_center_list.shape[0] != 0:
            center_pcd = npxyz_to_pcd(self.prob.cluster_center_list)
            center_pcd_kdtree = o3d.geometry.KDTreeFlann(center_pcd)
            k, idx, _ = center_pcd_kdtree.search_knn_vector_3d(
                mesh_pcd_center, 1
            )
            p = center_pcd.select_by_index(idx)
            for i in idx:
                if (
                    np.linalg.norm(
                        self.prob.cluster_center_list[i] - mesh_pcd_center
                    )
                    < 0.5
                ):
                    list_0.append(i)
            print("STEP1 list_0: ", list_0)
            if list_0 == []:
                v_cluster = np.prod(
                    self.prob.patch_recover[idx[0]].get_max_bound()
                    - self.prob.patch_recover[idx[0]].get_min_bound()
                )
                v_mesh = np.prod(
                    self.mesh_pcd.get_max_bound()
                    - self.mesh_pcd.get_min_bound()
                )
                # o3d.visualization.draw_geometries([self.prob.patch_recover[idx[0]], self.mesh_pcd])
                print(
                    "v_cluster, v_mesh: ", v_cluster, v_mesh, v_cluster / v_mesh
                )
                if v_cluster / v_mesh > 5:
                    min_bound = self.mesh_pcd.get_min_bound()
                    max_bound = self.mesh_pcd.get_max_bound()
                    for hv_id in [0, 1]:
                        sel = []
                        pts = self.prob.patch[hv_id].points
                        for i in range(len(pts)):
                            fitted = True
                            for j in [0, 1]:
                                if (
                                    pts[i][j] > max_bound[j] + self.TABU_RANGE
                                    or pts[i][j]
                                    < min_bound[j] - self.TABU_RANGE
                                ):
                                    fitted = False
                                    break
                            if fitted:
                                sel.append(i)

                        ori_pcd = self.prob.patch[hv_id]
                        ori_pcd.paint_uniform_color([1, 0, 0])

                        mesh_pcd = self.prob.patch[hv_id].select_by_index(sel)
                        mesh_pcd.paint_uniform_color([0, 1, 1])

                        # print('len(sel): ',len(sel))
                        # print(sel)
                        # print(self.prob.patch[hv_id])
                        self.prob.patch[hv_id] = self.prob.patch[
                            hv_id
                        ].select_by_index(sel, invert=True)
                        # print(self.prob.patch[hv_id])

                        self.prob.patch[hv_id].paint_uniform_color(
                            random_color()
                        )

                        # o3d.visualization.draw_geometries([mesh_pcd,ori_pcd,self.prob.patch[hv_id]])

                        self.prob.kdtree[hv_id] = o3d.geometry.KDTreeFlann(
                            self.prob.patch[hv_id]
                        )
                else:
                    list_0.append(idx[0])

            print("STEP2 list_0: ", list_0)
            if list_0 != []:
                fitted_pc = self.prob.patch_recover[list_0[0]]
                # o3d.visualization.draw_geometries([fitted_pc, self.mesh_pcd, p])
                fitted_pc = np.array(fitted_pc.points)
                del self.prob.patch_recover[list_0[0]]
                self.prob.cluster_center_list = np.delete(
                    self.prob.cluster_center_list, list_0[0], axis=0
                )

                for hv_id in [0, 1]:
                    sel = set()
                    for i in range(fitted_pc.shape[0]):
                        k, idx, _ = self.prob.kdtree[
                            hv_id
                        ].search_radius_vector_3d(fitted_pc[i], 0.1)
                        sel = sel.union(set(idx))
                    sel = list(sel)
                    print("len(sel): ", len(sel))
                    self.prob.patch[hv_id] = self.prob.patch[
                        hv_id
                    ].select_by_index(sel, invert=True)
                    obb = self.mesh_pcd.get_oriented_bounding_box()
                    obb_width = (obb.extent[0] + obb.extent[1]) / 2
                    obb_tabu_width = obb_width + self.TABU_RANGE * 2
                    obb.scale(
                        obb_tabu_width / obb_width, center=obb.get_center()
                    )
                    print("obb.extent: ", obb.extent)
                    self.prob.patch[hv_id] = self.prob.patch[hv_id].crop(
                        obb, invert=True
                    )
                    self.prob.kdtree[hv_id] = o3d.geometry.KDTreeFlann(
                        self.prob.patch[hv_id]
                    )
            # o3d.visualization.draw_geometries([self.prob.patch[0], self.prob.patch[1]])
            return True
        else:
            return False

    def load_mesh_file(self, fname: str, scale=None):
        mesh = o3d.io.read_triangle_mesh(fname, enable_post_processing=True)
        center = mesh.get_center()
        mesh.translate(self.prob.origin.get_center() - center)
        if scale is not None:
            mesh.scale(scale, [0, 0, 0])
        self.mesh = Mesh(mesh)
        self.mesh.center = mesh.get_center()
        if "column" in fname.lower():
            return ColumnComponent()
        elif "office" in fname.lower():
            return OfficeComponent()
        elif "room" in fname.lower():
            return OfficeComponent()
        elif "door" in fname.lower():
            return DoorComponent()
        elif "gong" in fname.lower():
            return DougongComponent()
        else:
            return None
        # mesh = o3d.io.read_triangle_mesh(fname, enable_post_processing = True)
        # mesh_v = mesh.getAssociatedCloud().points()
        # center = np.mean(mesh_v, axis=0)

        # trans = self.prob.origin.get_center()-center
        # mesh = pycc_translation(mesh, x=trans[0], y=trans[1], z=trans[2], scale_x=scale[0], scale_y=scale[1], scale_z=scale[2])

        # self.mesh = Mesh(mesh)
        # self.mesh.center = center
        # if self.window is not None:
        #   self.window.add_geometry(self.mesh.mesh, False)
        #   self.window.update_renderer()
        #   self.window.poll_events()

    def load_prob_file(self, fname: str):
        if os.path.isfile(self.pcd_cache_path) and os.path.samefile(
            self.pcd_cache_path, fname
        ):
            return
        self.pcd_cache_path = fname
        pcd = o3d.io.read_point_cloud(fname)
        # pcd.translate(-pcd.get_center())
        self.prob = PCD(pcd, cluster=True)

    # eval
    def evaluate(self, idx, patch_id, param, tmp_p=[0, 0, 0]):
        p = self.mesh.pcd.patch[patch_id].points[idx]
        c = param["c"]
        rz = param["rz"]
        if rz != 0:
            tmp_p[0] = p[0] * np.cos(rz) + p[1] * np.sin(rz) + c[0]
            tmp_p[1] = -p[0] * np.sin(rz) + p[1] * np.cos(rz) + c[1]
            tmp_p[2] = p[2] + c[2]
        else:
            for i in [0, 1, 2]:
                tmp_p[i] = p[i] + c[i]
        # print('self.prob.WEIGHT_NORNAL_ERROR: ',self.prob.WEIGHT_NORNAL_ERROR)
        if self.prob.WEIGHT_NORNAL_ERROR > 0:
            return self.prob.dist_normal(
                tmp_p, patch_id, self.mesh.pcd.patch[patch_id].normals[idx]
            )
        return self.prob.dist(tmp_p, patch_id)

    def fitness(self, x, nlopt_func_data=None):
        d = self.comp.decode_x(x)
        err_sum = 0.0
        num_sum = 0
        height = (
            self.mesh.mesh.get_max_bound()[2]
            - self.mesh.mesh.get_min_bound()[2]
        )
        # parallel ?
        w = self.comp.params["hv_weights"]
        tmp_p = [0, 0, 0]
        T1 = time.time()
        for i in [0, 1]:
            if w[i] > 0:
                err_a = 0.0
                num_a = 0
                points = self.mesh.pcd.patch[i].points
                num_pts = len(points)
                min_bound_z = self.mesh.pcd.patch[i].get_min_bound()[2]
                for idx in range(num_pts):
                    if "ignore_z" in self.comp.params:
                        if (points[idx][2] - min_bound_z) <= self.comp.params[
                            "ignore_z"
                        ] * height:
                            continue
                    v, idx = self.evaluate(idx, i, d, tmp_p)
                    err_a += v
                    num_a += 1
                if num_a > 0:
                    err_sum += err_a * w[i] * w[i]
                    num_sum += num_a * w[i]
        f = np.sqrt(err_sum / num_sum)
        T2 = time.time()
        self.iter += 1
        # if self.VERBOSE:
        #   print('self.iter, f, self.comp.params:',self.iter, f, self.comp.params['c'])
        # print('cluster_idx_0: ',self.cluster_idx_0)
        # print('cluster_idx_1: ',self.cluster_idx_1)

        if self.best_f is not None and self.best_f < 0.1:
            return self.best_f

        else:
            if (
                self.best_f is None
                or (not self.toMax and f < self.best_f)
                or (self.toMax and f > self.best_f)
            ):
                self.best_f = f
                self.best_x = np.array(x, copy=True)
                self.comp.params["best_c"] = self.comp.params["c"]
                self.comp.params["best_rz"] = self.comp.params["rz"]
                if self.VERBOSE:
                    print("\t\t***", self.best_f, self.best_x)
                    # print('current best idx: ', self.iter, idx, self.prob.patch_0[idx])
            return f
