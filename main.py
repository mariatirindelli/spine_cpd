import numpy as np
import pycpd
import cv2
import matplotlib.pyplot as plt

root = "C:/Users/maria/OneDrive/Desktop/test_pc/2D/"

def plot2D(X, Y, mux, muy):
    X = X * 1000
    Y = Y * 1000
    mux = mux * 1000
    muy = muy * 1000
    # 1 discretize the space in pixels
    radius = 15
    spacing = 0.1

    print(mux, "  ", muy)

    tl_x = min(np.min(X[:, 0]), np.min(Y[:, 0]), mux[0], muy[0]) - radius*spacing
    tl_y = min(np.min(X[:, 1]), np.min(Y[:, 1]), mux[1], muy[1]) - radius*spacing

    tr_x = max(np.max(X[:, 0]), np.max(Y[:, 0]), mux[0], muy[0]) + radius*spacing
    tr_y = max(np.max(X[:, 1]), np.max(Y[:, 1]), mux[1], muy[1]) + radius*spacing

    image_width_phys = tr_x - tl_x
    image_height_phys = tr_y - tl_y

    image_width = int(image_width_phys/spacing) + 1
    image_height = int(image_height_phys/spacing) + 1

    image = np.zeros((image_height, image_width))

    mu_x_pix = (int( (mux[0] - tl_x)/spacing), int( (mux[1] - tl_y)/spacing))
    cv2.circle(image, mu_x_pix, 7, color=(1), thickness=-1)

    mu_y_pix = (int( (muy[0] - tl_x)/spacing), int( (muy[1] - tl_y)/spacing))
    cv2.circle(image, mu_y_pix, 5, color=(2), thickness=-1)

    for i in range(X.shape[0]):
        X_i = (int( (X[i, 0] - tl_x)/spacing), int( (X[i, 1] - tl_y)/spacing))
        cv2.circle(image, X_i, 3, color=(1), thickness = -1)
    for j in range(Y.shape[0]):
        Y_i = (int((Y[j, 0] - tl_x) / spacing), int((Y[j, 1] - tl_y) / spacing))
        cv2.circle(image, Y_i, 3, color=(2), thickness = -1)

    plt.imshow(image)




# def plot2D(X, Y, mux=0, muy=0):
#     # 1 discretize the space in pixels
#     radius = 5
#     spacing = 0.1
#
#     tl_x = min(np.min(X[:, 0]), np.min(Y[:, 0])) - radius*spacing
#     tl_y = min(np.min(X[:, 1]), np.min(Y[:, 1])) - radius*spacing
#
#     tr_x = max(np.max(X[:, 0]), np.max(Y[:, 0])) + radius*spacing
#     tr_y = max(np.max(X[:, 1]), np.max(Y[:, 1])) + radius*spacing
#
#     image_width_phys = tr_x - tl_x
#     image_height_phys = tr_y - tl_y
#
#
#     image_width = int(image_width_phys/spacing) + 1
#     image_height = int(image_height_phys/spacing) + 1
#
#     for i in range(X.shape[0]):
#         image = np.zeros((image_height, image_width))
#         X_i = (int( (X[i, 0] - tl_x)/spacing), int( (X[i, 1] - tl_y)/spacing))
#         cv2.circle(image, X_i, 3, color=(1), thickness = -1)
#         for j in range(Y.shape[0]):
#             Y_i = (int((Y[j, 0] - tl_x) / spacing), int((Y[j, 1] - tl_y) / spacing))
#             cv2.circle(image, Y_i, 3, color=(2+j), thickness = -1)
#
#         plt.imshow(image)
#         plt.show()


def main(root_dir):

    source1 = np.loadtxt(root_dir + "source_c1_pre_reg.txt")[:, 0:3]/1000  # M1x3 point cloud
    source2 = np.loadtxt(root_dir + "source_c2.txt")[:, 0:3]/1000  # M2x3 point cloud
    target = np.loadtxt(root_dir + "target_full.txt")[:, 0:3]/1000  # Nx3 point cloud

    # for alpha in range(10, 100, 10):
    #     cpd_method = BiomechanicalCpd(target, source1, spring_indexes=[0], alpha=alpha)
    #     cpd_method.expectation()
    #     mux, muy = cpd_method.get_means()
    #     plot2D(cpd_method.X, cpd_method.Y, mux, muy)

    cpd_method = BiomechanicalCpd(target, source1, spring_indexes=[1], alpha=2**4, sigma=1)
    plt.ion()
    for iter in range(100):
        print("iter: " + str(iter))

        cpd_method.expectation()
        plt.show()
        plt.pause(1)
        cpd_method.maximization()


    #cpd_method.expectation()



main(root)