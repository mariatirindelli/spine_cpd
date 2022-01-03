from copy import deepcopy
import meshio

class MyClass:
    def __init__(self, my_list=[]):
        self.my_list2 = my_list
        self.my_list = deepcopy(my_list)

    def print(self):
        print(self.my_list)
        print(self.my_list2)


a = MyClass()
b = MyClass()

a.my_list.append(3)
a.my_list2.append(3)
b.print()
a.print()


translation = 30
initial_mesh = meshio.read("E:/NAS/jane_project/obj_files/spine10/ts_1_0/spine10_vert1_1_0.obj")

translated_points = initial_mesh.points.copy()
translated_points[:, 0] = translated_points[:, 0] + translation  # apply translation on x direction

mesh = meshio.Mesh(
                    translated_points,
                    initial_mesh.cells,
                    # Optionally provide extra data on points, cells, etc.
                    initial_mesh.point_data,
                    # Each item in cell data must match the cells array
                    initial_mesh.cell_data,
                    )

mesh.write("C:/Users/maria/OneDrive/Desktop/pc_results/tmp.obj")