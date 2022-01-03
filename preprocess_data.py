"""
The script pre-process data to bring them to a reasonable format
"""
import numpy as np
from utils import Spring

def order_connection(item, vertebral_level_idxes):
    """
    Given  the connection item = (connection_index_1, connection_index_2), indicating a connection between two points
    (indexes) in a point cloud, the function first detects which of the connection points node belongs to the input
    vertebra (i.e. which index is contained in vertebral_level_idxes indicating the indexes of the points in the cloud
    belonging to a given vertebra). If the first point in the tuple is the one belonging to the input vertebra, the
    function returns the input item with the same order. Otherwise, it returns the input item with swapped elements,
    in a way that the first element in the return item (connection) is always the point belonging to the input
    vertebra
    """
    if item[0] in vertebral_level_idxes:
        return item

    return item[1], item[0]

def get_vertebra_id(vertebral_level):
    return "vert_" + vertebral_level + 1

def get_springs_from_vertebra(vertebral_level_idxes, constraints_pairs):
    """
    It returns the list of connection starting from the input vertebral level as a list of tuples like:
    [(idx_current_vertebra_level_0, idx_connected_vertebra_level_1),
    (idx_current_vertebra_level_1, idx_connected_vertebra_level_2),
                                ...,
    (idx_current_vertebra_level_n, idx_connected_vertebra_level_n)]
    """
    current_vertebra_springs = [item for item in constraints_pairs if item[0] in vertebral_level_idxes
                                or item[1] in vertebral_level_idxes]

    current_vertebra_springs = [order_connection(item, vertebral_level_idxes) for item in current_vertebra_springs]

    springs = [Spring(current_id = , next_id, s_current, s_next)]

    return current_vertebra_springs


def preprocess_input(source_pc, gt_flow, position1, tre_points):

    vertebra_dict = []

    for i, vertebral_level_idxes in enumerate(position1):

        # 2.a Extracting the points belonging to the first vertebra
        current_vertebra = source_pc[vertebral_level_idxes, ...]
        current_flow = gt_flow[vertebral_level_idxes, ...]

        # 2.b Getting all the springs connections starting from the current vertebra
        current_vertebra_springs = get_springs_from_vertebra(vertebral_level_idxes, constrain_pairs)

        # 2.3 Generating the pairs: (current_vertebra_idx, constraint_position) where current_vertebra_idx
        # is the spring connection in the current_vertebra object and constraint_position is the position ([x, y, z]
        # position) of the point connected to the spring
        current_vertebra_connections = [(np.argwhere(vertebral_level_idxes == item[0]), source_pc[item[1]])
                                        for item in current_vertebra_springs]

        gt_T = get_gt_transform(source_pc=current_vertebra,
                                gt_flow=current_flow)

        tre_point = tre_points[tre_points[:, -1] == i+1, :]

        vertebra_dict.append({'source': current_vertebra,
                              'gt_flow': current_flow,
                              'springs': current_vertebra_connections,
                              'gt_transform': gt_T,
                              'tre_points': tre_point})

    return vertebra_dict











# ##############################################################################################################
# ############################################## Getting the data ##############################################
# ##############################################################################################################
source_pc, target_pc, color1, color2, gt_flow, mask1, constraint, position1, position2, file_name, tre_points \
    = data_batch

constrain_pairs = get_connected_idxes(constraint)
for i, item in enumerate(constrain_pairs):
    save_data(data_dict={'constraint_' + str(i): source_pc[item, ...]},
              save_path=os.path.join(save_path, file_name))

# Preprocessing and saving unprocessed data
vertebra_dict = preprocess_input(source_pc, gt_flow, position1, constrain_pairs, tre_points)

# ##############################################################################################################
# ################################ 1.  1st CPD iteration on the full spine #####################################
# ##############################################################################################################

# 1.a First iteration to alight the spines
cpd_method = BiomechanicalCpd(target_pc=target_pc, source_pc=source_pc, max_iterations=cpd_iterations)

try:
    source_pc_it1, predicted_T_it1 = run_registration(cpd_method, with_callback=plot_iterations)