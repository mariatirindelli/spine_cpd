#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import pandas as pd
from itertools import chain
from utils import Spring

# Part of the code is referred from: https://github.com/charlesq34/pointnet

def get_random_rotation():
    angle_target = np.random.randint(-20, 20)

    xyz = np.random.choice(["x", "y", "z"])

    r = Rotation.from_euler(xyz, angle_target * np.pi / 180)
    return r.as_matrix()


def apply_random_rotation(pc, rotation_center=np.array([0, 0, 0]), r=None):

    r = get_random_rotation() if r is None else r
    pc = pc - rotation_center
    rotated_pc = np.dot(pc, r)

    return rotated_pc + rotation_center


def augment_test(flow, pc1, pc2, tre_points, rotation, axis):
    # ###### Generating the arrays where to store the augmented data - the fourth dimension remains constant #######
    augmented_pc1 = np.zeros(pc1.shape)
    augmented_pc2 = np.zeros(pc2.shape)

    if pc1.shape[1] == 4:
        augmented_pc1[:, -1] = pc1[:, -1]
        pc1 = pc1[:, :3]

    if pc2.shape[1] == 4:
        augmented_pc2[:, -1] = pc2[:, -1]
        pc2 = pc2[:, :3]

    angle_target = rotation
    xyz = axis

    # ###### Augmenting the data #######
    # The ground truth position of the deformed source
    gt_target = pc1 + flow

    # rotate the source
    r = Rotation.from_euler(xyz, angle_target * np.pi / 180)
    r = r.as_matrix()
    pc1 = apply_random_rotation(pc1, r=r, rotation_center=np.mean(pc1, axis=0))
    tre_points[:, 0:3] = apply_random_rotation(tre_points[:, 0:3], r=r, rotation_center=np.mean(pc1, axis=0))

    # recompute the flow with the updated pc1 and gt_target
    flow = gt_target - pc1

    augmented_pc1[:, 0:3] = pc1
    augmented_pc2[:, 0:3] = pc2

    return flow, augmented_pc1, augmented_pc2, tre_points


def augment_data(flow, pc1, pc2, tre_points, augmentation_prob=0.5):
    # ###### Generating the arrays where to store the augmented data - the fourth dimension remains constant #######
    augmented_pc1 = np.zeros(pc1.shape)
    augmented_pc2 = np.zeros(pc2.shape)

    if pc1.shape[1] == 4:
        augmented_pc1[:, -1] = pc1[:, -1]
        pc1 = pc1[:, :3]

    if pc2.shape[1] == 4:
        augmented_pc2[:, -1] = pc2[:, -1]
        pc2 = pc2[:, :3]

    # ###### Augmenting the data #######
    # The ground truth position of the deformed source
    gt_target = pc1 + flow

    # rotate the source with a probability 0.5
    if np.random.random() < augmentation_prob:
        # rotate the source about its centroid randomly and update flow accordingly

        r = get_random_rotation()
        pc1 = apply_random_rotation(pc1, r=r, rotation_center=np.mean(pc1, axis=0))
        tre_points[:, 0:3] = apply_random_rotation(tre_points[:, 0:3], r=r, rotation_center=np.mean(pc1, axis=0))

    # rotate the target with a probability 0.5
    if np.random.random() < augmentation_prob:
        # apply the same rotation to ground truth target and pc2 (a rotation about the target centroid)
        # and update flow accordingly
        r = get_random_rotation()
        pc2 = apply_random_rotation(pc2, r=r, rotation_center=np.mean(pc2, axis=0))
        gt_target = apply_random_rotation(gt_target, r=r, rotation_center=np.mean(pc2, axis=0))

    # recompute the flow with the updated pc1 and gt_target
    flow = gt_target - pc1

    # add noise to the target points with a probability of 0.5
    if np.random.random() < augmentation_prob:
        pc2 = pc2 + np.random.normal(0, 2, pc2.shape)

    augmented_pc1[:, 0:3] = pc1
    augmented_pc2[:, 0:3] = pc2

    return flow, augmented_pc1, augmented_pc2, tre_points


def read_numpy_file(fp):
    data = np.load(fp)
    pos1 = data["pc1"].astype('float32')
    pos2 = data["pc2"].astype('float32')
    flow = data["flow"].astype('float32')
    constraint = data["ctsPts"].astype('int')
    return constraint, flow, pos1, pos2


def _get_spine_number(path: str):
    name = os.path.split(path)[-1]
    name = name.split("ts")[0]
    name = name.replace("raycasted", "")
    name = name.replace("full", "")
    name = name.replace("spine", "")
    name = name.replace("_", "")
    try:
        return int(name)
    except:
        return -1


class SceneflowDataset(Dataset):
    def __init__(self, npoints=4096, root='/mnt/polyaxon/data1/Spine_Flownet/raycastedSpineClouds/', mode="test",
                 raycasted = True, augment=False, data_seed=0, use_target_normalization_as_feature = True, **kwargs):
        """
        :param npoints: number of points of input point clouds
        :param root: folder of data in .npz format
        :param mode: mode can be any of the "train", "test" and "validation"
        :param raycasted: the data used is raycasted or full vertebras
        :param raycasted: the data used is raycasted or full vertebrae
        :param augment: if augment data for training
        :param data_seed: which permutation to use for slicing the dataset
        """

        if mode not in ["train", "val", "test"]:
            raise Exception(f'dataset mode is {mode}. mode can be any of the "train", "test" and "validation"')

        self.npoints = npoints
        self.mode = mode
        self.root = root
        self.raycasted = raycasted
        self.augment = augment
        self.data_path = glob.glob(os.path.join(self.root, '*.npz'))
        self.data_path = [item for item in self.data_path if "ts_20" in item]
        self.use_target_normalization_as_feature = use_target_normalization_as_feature

        self.spine_splits = {"test": np.arange(1, 23)}
        self.data_path = [path for path in self.data_path if _get_spine_number(path) in self.spine_splits[self.mode]]

        if "augment_test" in kwargs.keys() and kwargs["augment_test"]:
            self.augment_test = kwargs["augment_test"]
            self.test_rotation_degree = kwargs["test_rotation_degree"]
            self.test_rotation_axis = kwargs["test_rotation_axis"]
        else:
            self.augment_test = False
            self.test_rotation_degree = None
            self.test_rotation_axis = None

    def get_tre_points(self, filename):
        """
        Loading the points position for TRE error computation in testing. They are saved in the same folder as the
        data as spine_id + "_facet_targets.txt".
        :param filename: The input filename
        """
        # Example: filename = some_fold/raycasted_spine22_ts_7_0.npz

        # --> filename = raycasted_spine22_ts_7_0.npz
        filename = os.path.split(filename)[-1]

        # --> spine_id = spine22
        spine_id = [item for item in filename.split("_") if "spine" in item][0]

        # todo: remove this in future, only for wrongly named data
        spine_id = spine_id.replace("raycasted", "")
        spine_id = spine_id.replace("ts", "")

        # --> target_points_filepath = self.root/spine22_facet_targets.txt
        target_points_filepath = os.path.join(self.root, spine_id + "_facet_targets.txt")

        return np.loadtxt(target_points_filepath)

    def get_downsampled_idx(self, pc, random_seed, constraints=None, sample_each_vertebra=True):

        """
        :param pc: [Nx4] input point cloud, where:
            pc[i, 0:3] = (x, y, z) positions of the i-th point of the source point cloud
            pc[i, 4] = integer indicating the vertebral level the point i-th of the input point cloud belongs to

        :param random_seed: The random seed to search the random sample of points

        :param constraints: list of constraints idxs. Currently it is like:
            [L1.1, L2.1, L2.2, L3.1, L3.2, L4.1, L4.2, L5.1] where Lx.i is the i-th constraint point, lying on
            vertebra x

        :param sample_each_vertebra: A boolean that indicates if the sampling must be done separately for each vertebra,
            assuming that vertebral levels are indicated on the 4th column of the input point cloud (pc).
            If set to True, the script samples self.npoints/5 points from each vertebra

        :return The indexes of the input pc to be used to downsample the point cloud.
        """

        if constraints is not None and not sample_each_vertebra:
            raise NotImplementedError("Constraints are not supported if sample_each_vertebra is False")

        # 1. Down-sample the point cloud
        np.random.seed(random_seed)

        if sample_each_vertebra:

            # 1.a) L1, L2, L3, L4, L5 = indexes of vertebra 1, 2, 3, 4, 5
            # sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5 = down-sampled indexes of vertebra
            L1, L2, L3, L4, L5, sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5 = self.sample_vertebrae(
                pc)

            # 1.b) Concatenating the all the points together
            sample_idx_ = np.concatenate((sample_idx1, sample_idx2,
                                          sample_idx3, sample_idx4,
                                          sample_idx5), axis=0).astype(int)

        else:
            sample_idx_ = np.random.choice(pc, self.npoints, replace=False)
            return sample_idx_

        if constraints is None:
            return sample_idx_

        # 2. If constraints are also passed, then make space for the constraint points in the sample_idx_
        # points which will be deleted from the source point indexes to make space for the constraints

        # 2.a) Removing K points from the point cloud, with K = N constraints
        pc_lengths = [item.size for item in [sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5]]
        pc_idx_centers = [np.sum(pc_lengths[0:i + 1]) - pc_lengths[i] // 2 for i in range(5)]
        constraints_per_vertebra = [len(np.where(pc[constraints, -1] == i)[0]) for i in range(1, 6)]
        delete_list = list(chain(*[range(center, center + item)
                                   for (center, item) in zip(pc_idx_centers, constraints_per_vertebra)]))
        sample_idx_ = np.delete(sample_idx_, delete_list)

        # 2.b) Adding the constraints points indexes in the end.
        sample_idx_ = np.concatenate((sample_idx_, constraints), axis=0).astype(int)

        # 2.c) As we have concatenated the constraints indexes at the end of the indexes array, the position of the
        # constraints in the downsampled point cloud will be at the end of it
        updated_constraints_idx = [i for i in range(len(sample_idx_) - len(constraints), len(sample_idx_))]

        return sample_idx_, updated_constraints_idx

    @staticmethod
    def get_centroid(input_pc):
        """
        :param input_pc: [Nx3] array of points
        """

        assert input_pc.shape[1] == 3
        centroid = np.mean(input_pc, axis=0)

        return centroid

    def normalize_data(self, source_pc, target_pc, tre_points = None):
        """
        The function normalizes the data according to the Fu paper:

        Given
        - vs_c = source centroid
        - vt_c = target centroid

        - vs_i = i-th point in the source point cloud
        - vt_i = i-th point in the target point cloud

        vs_i_norm = [vs_i - vs_c, vs_i - vt_c, label] =
        = [vs_i.x-vs_c.x, vs_i.y-vs_c.y, vs_i.z-vs_c.z, vs_i.x-vt_c.x, vs_i.y-vt_c.y, vs_i.z-vt_c.z, vs_i_label]

        vt_i_norm = [vt_i - vs_c, vt_i - vt_c, label] =
        = [vt_i.x-vs_c.x, vt_i.y-vs_c.y, vt_i.z-vs_c.z, vt_i.x-vs_c.x, vt_i.y-vs_c.y, vt_i.z-vs_c.z, vt_i_label]

        :param source_pc: [Nx3] array containing the coordinates and vertebra level of the source point cloud.
            Specifically: source_pc[i, 0:3] = (x, y, z) positions of the i-t point of the source point cloud

        :param target_pc: [Nx3] array containing the coordinates and vertebra level of the target point cloud.
            Specifically: target_pc[i, 0:3] = (x, y, z) positions of the i-t point of the target point cloud

        :param tre_points: Additional points to be transformed (e.g. to get the TRE error). This are only normalized
            wrt the source centroid
        """

        assert source_pc.shape[1] == target_pc.shape[1] == 3, "Input point clouds must have shape Nx3"

        vs_c = self.get_centroid(source_pc)  # source centroid
        vt_c = self.get_centroid(target_pc)  # target centroid

        vs_normalized = np.concatenate((source_pc - vs_c, source_pc - vt_c), axis=1)
        vt_normalized = np.concatenate((target_pc - vs_c, target_pc - vt_c), axis=1)

        if tre_points is not None:
            tre_points[:, 0:3] = tre_points[:, 0:3] - vs_c

        return vs_normalized, vt_normalized, tre_points

    def get_constraints(self, file_id):
        spine_id = _get_spine_number(file_id)
        filename = os.path.join(self.root, "spine" + str(spine_id) + "_constraints.csv")
        df = pd.read_csv(filename)

        constraints = []
        for i in range(df.shape[0]):
            position = np.array([df.loc[i, "x"], df.loc[i, "y"], df.loc[i, "z"]])
            constraints.append(Spring(start_id=df.loc[i, "node1"],
                                      end_id=df.loc[i, "node2"],
                                      position=np.reshape(position, (1, 3))))
        return constraints

    def __getitem__(self, index):

        file_id = os.path.split(self.data_path[index])[-1].split(".")[0]

        tmp, flow, source_pc, target_pc = read_numpy_file(fp=self.data_path[index])

        # spurious code to retrieve constraints
        # print("\n----")
        # print(file_id)
        # print(source_pc[tmp, :])
        # return source_pc[tmp, :], file_id

        # Loading the constraint in the new way they're saved
        constraints = self.get_constraints(file_id)

        # Getting the indexes to down-sample the source and target point clouds and the updated constraints indexes
        sample_idx_source = \
            self.get_downsampled_idx(pc=source_pc, random_seed=100, constraints=None, sample_each_vertebra=True)
        sample_idx_target = self.get_downsampled_idx(pc=target_pc, random_seed=20, sample_each_vertebra=True)

        # Down-sampling the point clouds
        downsampled_source_pc = source_pc[sample_idx_source, ...]
        downsampled_target_pc = target_pc[sample_idx_target, ...]
        downsampled_flow = flow[sample_idx_source, :]

        tre_points = self.get_tre_points(self.data_path[index])

        # augmentation in train
        if self.mode == "train" and self.augment:
            downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points = \
                augment_data(downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points,
                             augmentation_prob=0.5)

        if self.mode == "test" and self.augment_test:
            downsampled_flow, downsampled_source_pc, downsampled_target_pc, tre_points = \
                augment_test(flow=downsampled_flow,
                             pc1=downsampled_source_pc,
                             pc2=downsampled_target_pc,
                             tre_points=tre_points,
                             rotation=self.test_rotation_degree,
                             axis=self.test_rotation_axis)

        # For CPD the normalization is not needed

        return downsampled_source_pc, downsampled_target_pc, downsampled_flow, constraints, file_id, tre_points

    def sample_vertebrae(self, pos1):

        # dividing by the number of vertebrae
        n_points = self.npoints // 5

        surface1 = np.copy(pos1)[:, 3]
        # specific for vertebrae: sampling 4096 points
        L1 = np.argwhere(surface1 == 1).squeeze()
        sample_idx1 = np.random.choice(L1, n_points, replace=False)
        L2 = np.argwhere(surface1 == 2).squeeze()
        sample_idx2 = np.random.choice(L2, n_points, replace=False)
        L3 = np.argwhere(surface1 == 3).squeeze()
        sample_idx3 = np.random.choice(L3, n_points, replace=False)
        L4 = np.argwhere(surface1 == 4).squeeze()
        sample_idx4 = np.random.choice(L4, n_points, replace=False)
        L5 = np.argwhere(surface1 == 5).squeeze()
        sample_idx5 = np.random.choice(L5, n_points, replace=False)
        return L1, L2, L3, L4, L5, sample_idx1, sample_idx2, sample_idx3, sample_idx4, sample_idx5

    def __len__(self):
        return len(self.data_path)
