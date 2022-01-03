import numpy as np
import cv2

class Spring:
    def __init__(self, start_id, end_id, position):

        self.start_id = start_id
        self.end_id = end_id
        self.position = position


def extract_pc_plane(pc, plane_position, plane_ax, plane_thickness = 3):

    if plane_ax == 0:
        show_points = pc[np.abs(pc[:, 0] - plane_position) < plane_thickness, :]
        return show_points[:, 1::]

    elif plane_ax == 1:
        show_points = pc[np.abs(pc[:, 1] - plane_position) < plane_thickness, :]
        return show_points[:, [0, 2]]

    elif plane_ax == 2:
        show_points = pc[np.abs(pc[:, 2] - plane_position) < plane_thickness, :]
        return show_points[:, [1, 2]]
    else:
        raise NotImplementedError


def get_image_properties(*args, radius=3):

    all_x = np.concatenate([item[:, 0] for item in args])
    all_y = np.concatenate([item[:, 1] for item in args])

    tl_x = np.min(all_x) - radius
    tl_y = np.min(all_y) - radius

    br_x = np.max(all_x) + radius
    br_y = np.max(all_y) + radius

    image_width = br_x - tl_x
    image_height = br_y - tl_y

    return (tl_x, tl_y), image_height, image_width


def initialize_image(physical_height, physical_width, spacing):
    pixel_height = int(physical_height/spacing)
    pixel_width = int(physical_width/spacing)

    return np.zeros((pixel_height, pixel_width))


def add_pc2image(pc, image, image_translation, spacing, radius=5, color=1):
    pc[:, 0] = pc[:, 0]-image_translation[0]
    pc[:, 1] = pc[:, 1] - image_translation[1]
    pc = pc/spacing
    pc = pc.astype(np.int)

    image[pc[:, 1], pc[:, 0]] = color

    kernel = np.ones((radius, radius), 'uint8')
    image = cv2.dilate(image, kernel, iterations=1)

    return image

