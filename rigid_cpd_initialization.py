import numpy as np
from data import SceneflowDataset
from typing import List, Tuple
from utils import Spring
from BiomechanicCPD import BiomechanicalCpd
from copy import deepcopy
import meshio
import os
from sklearn.neighbors import KDTree
import argparse

class Vertebra:
    def __init__(self, points, next_vertebra_connections = [], previous_vertebra_connections = [], R=np.eye(3), t=0):
        self._points = points
        self._N = self._points.shape[0]
        self.next_vertebra_connections = deepcopy(next_vertebra_connections)
        self.previous_vertebra_connections = deepcopy(previous_vertebra_connections)
        self.R = R
        self.t = t

    def add_point(self, point):

        self._points = np.concatenate( (self._points, point), axis = 0)
        self._N = self._points.shape[0]

    def num_points(self):
        return self._N

    def points(self, apply_transform=False):
        if apply_transform:
            raise NotImplementedError
        return self._points

    def at(self, idx, apply_transform=False):
        point = np.reshape(self._points[idx, :], (1, 3))
        if apply_transform:
            return np.reshape(np.dot(point, self.R) + self.t, (1, 3))
            
        return self._points[idx, :]

def save_data(save_dir, *args):
    for arg in args:
        save_path = os.path.join(save_dir, arg.__name__ + ".txt")
        np.savetxt(save_path, arg)

def get_closest_points(pc1, pc2):
    """
    returns the points of pc1 which are closest to pc2
    """
    kdtree=KDTree(pc1[:,:3])
    dist, ind =kdtree.query(pc2[:,:3], 1)
    ind = ind.flatten()
    points = pc1[ind, ...]

    return points



def split_source_and_add_constraints(source_data, constraints: List[Spring]=[]):

    vertebrae_dict = {vertebral_level: Vertebra(points = source_data[source_data[:, -1] == vertebral_level, 0:3])
                        for vertebral_level in range(1, 6)}

    if len(constraints) == 0:
        return vertebrae_dict

    for i in range(1, 5):

        # getting connections starting from this vertebra
        starting_connections = [item for item in constraints if item.start_id == i]

        # Adding the point to the current vertebra
        for connection in starting_connections:

            # Adding the point to the current vertebra
            vertebrae_dict[i].add_point(connection.position)

            # Adding the point to the next vertebra
            vertebrae_dict[i + 1].add_point(connection.position)

            # Updating the next_verterba connection list in the current vertebra
            vertebrae_dict[i].next_vertebra_connections.append(
                (vertebrae_dict[i].num_points() - 1, vertebrae_dict[i + 1].num_points() - 1) )

            # Updating the previous_verterba connection list in the next vertebra
            vertebrae_dict[i + 1].previous_vertebra_connections.append(
                (vertebrae_dict[i + 1].num_points() - 1, vertebrae_dict[i].num_points() - 1))

    return vertebrae_dict


def get_springs(vertebrae_dict, i):
    """
    :param vertebrae_dict:
    :param i: vertebral level
    :return: a list of springs that contains tuple, where each tuple is defined as
    (current_idx, endpoint_position) where the currecnt_idx is the index where the spring is connected in the
    current vertebral level i and enpoint position is the position of the endpoint of the spring
    """
    springs = []
    for j in range(len(vertebrae_dict[i].previous_vertebra_connections)):
        if i != 1:
            springs.extend(
                [(vertebrae_dict[i].previous_vertebra_connections[j][0],  # The index of the spring in the current PC
                  vertebrae_dict[i - 1].at(vertebrae_dict[i].previous_vertebra_connections[j][1], apply_transform=True)
                  )])  # The position of the spring endpoint in the previous (transformed) PC

    for j in range(len(vertebrae_dict[i].next_vertebra_connections)):
        if i != 5:
            springs.extend([(vertebrae_dict[i].next_vertebra_connections[j][0],  # The index of the spring in the current PC
                             vertebrae_dict[i + 1].at(vertebrae_dict[i].next_vertebra_connections[j][1],
                                                      apply_transform=True)
                             )])  # The position of the spring endpoint in the previous (transformed) PC
    return springs

def transform_constraints(constraints, R, t):
    for i, _ in enumerate(constraints):
        point = np.reshape(constraints[i].position, (1, 3))
        transformed_point = np.dot(point, R) + t
        constraints[i].position = np.reshape(transformed_point, (1, 3))
        
    return constraints

def save_data(save_dir, pc):
    np.savetxt(save_dir, pc)

def save_mesh(original_mesh_dir, R, t, save_dir):
    initial_mesh = meshio.read(original_mesh_dir)

    transformed_points = np.dot(initial_mesh.points, R) + t

    mesh = meshio.Mesh(
        transformed_points,
        initial_mesh.cells,
        # Optionally provide extra data on points, cells, etc.
        initial_mesh.point_data,
        # Each item in cell data must match the cells array
        initial_mesh.cell_data,
    )
    mesh.write(save_dir)


def save_springs_points(cpd_model:BiomechanicalCpd, save_dir):
    num_springs = len(cpd_model.spring_indexes)
    x_springs = cpd_model.X[-num_springs::]
    y_springs = cpd_model.TY[cpd_model.spring_indexes, :]

    assert x_springs.shape[0] == y_springs.shape[0]
    save_dir = save_dir.replace(".txt", "")
    concatenated_points = np.concatenate((x_springs, y_springs), axis=0)
    np.savetxt(save_dir + "_spring" + ".txt", concatenated_points)


def process_data(data_batch, alpha, sigma, max_iterations=30, args=None):

    # Getting data
    source_pc, target_pc, flow, constraints, file_id, tre_points = data_batch
    print("Processing ", file_id)

    spine_id = file_id.split("_")[1]
    save_dir = os.path.join(args.save_dir, spine_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Running the first iteration of the rigid CPD
    cpd_method = BiomechanicalCpd(X=target_pc[:, 0:3],
                                  Y=source_pc[:, 0:3],
                                  max_iterations=30)

    TY, (_, R_reg, t_reg) = cpd_method.register(
                                                # callback=cpd_method.save_reg_results
                                               )
    save_data(os.path.join(save_dir, file_id + "source.txt"), source_pc[:, 0:3])
    save_data(os.path.join(save_dir, file_id + "global_cpd.txt"), TY)

    # Transform constraint points
    source_pc_transformed = np.zeros( (TY.shape[0], 4) )
    source_pc_transformed[:, 0:3] = TY
    source_pc_transformed[:, -1] = source_pc[:, -1]

    # Splitting vertebrae for the next groupwise transformation
    if args.use_springs:
        vertebrae_dict = split_source_and_add_constraints(source_pc, constraints)
    else:
        vertebrae_dict = split_source_and_add_constraints(source_pc)

    for i in range(1, 6):
        vertebrae_dict[i].R = R_reg
        vertebrae_dict[i].t = t_reg

    biomechanical_transformations = {}
    for i in range(1, 6):

        if args.use_springs:
            springs = get_springs(vertebrae_dict, i)
        else:
            springs = []

        if args.use_closest_points:
            X = get_closest_points(target_pc[:, 0:3], source_pc_transformed[source_pc_transformed[:, -1] == i, 0:3 ])
        else:
            X = target_pc[:, 0:3].copy()

        biomechanical_transformations[i] = BiomechanicalCpd(X=X,
                                                            Y=vertebrae_dict[i].points(apply_transform=False),
                                                            springs = springs,
                                                            alpha=alpha,
                                                            sigma=sigma,
                                                            R=R_reg,
                                                            t=np.reshape(t_reg, (1, 3)),
                                                            fix_variance=args.fix_variance)

        biomechanical_transformations[i].transform_point_cloud()

    for iteration in range(max_iterations):
        for i in range(1, 6):
            if args.use_springs:
                springs = get_springs(vertebrae_dict, i)
                biomechanical_transformations[i].update_springs(springs)

            biomechanical_transformations[i].expectation()
            biomechanical_transformations[i].maximization()

            vertebrae_dict[i].R = biomechanical_transformations[i].R
            vertebrae_dict[i].t = biomechanical_transformations[i].t

            if iteration % 10 == 0:

                save_data(os.path.join(save_dir,  file_id + "_target" + "_vert" + str(i) + ".txt"),
                          biomechanical_transformations[i].X)

                iter_save_dir = os.path.join(save_dir, "iter" + str(iteration))

                if not os.path.exists(iter_save_dir):
                    os.makedirs(iter_save_dir)

                save_data(os.path.join(iter_save_dir, file_id + "_vert" + str(i) + "_iter" + str(iteration) + ".txt"),
                          biomechanical_transformations[i].TY)

                mesh_dir = os.path.join(args.mesh_dir, spine_id, "ts0", spine_id + "_vert" + str(i) + "0.obj")

                save_mesh(mesh_dir, vertebrae_dict[i].R, vertebrae_dict[i].t,
                          os.path.join(iter_save_dir, file_id + "_vert" + str(i) + "_iter" + str(iteration) + ".obj"))

                if args.use_springs:
                    save_springs_points(biomechanical_transformations[i],
                                        os.path.join(iter_save_dir, file_id + "_vert" + str(i) + "_iter" + str(iteration) + ".txt"))

                print("")


def main():
    parser = argparse.ArgumentParser(description='Data generation testing')
    parser.add_argument('--dataset_path', type=str,
                        default="E:/NAS/jane_project/flownet_data/nas_data/new_data_raycasted")
    parser.add_argument('--save_dir', type=str, default="C:/Users/maria/OneDrive/Desktop/pc_results")
    parser.add_argument('--mesh_dir', type=str, default="E:/NAS/jane_project/obj_files")
    parser.add_argument('--fix_variance', action='store_true')
    parser.add_argument('--use_closest_points', action='store_false')
    parser.add_argument('--use_springs', action='store_true')
    parser.add_argument('--cpd-iterations', type=int, default=22)

    args = parser.parse_args()

    dataset_path = args.dataset_path
    test_set = SceneflowDataset(mode="test",
                                root=dataset_path,
                                raycasted=True
                                )

    for i, data in enumerate(test_set):
        process_data(data_batch=data,
                     alpha = 2**5,
                     sigma=1,
                     max_iterations=args.cpd_iterations,
                     args=args)

main()
# bool: update sigma (yes/no)
# bool subsample X (yes/no)


