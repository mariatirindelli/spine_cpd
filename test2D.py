import random

import numpy as np
import pycpd
import cv2
import matplotlib.pyplot as plt

root = "C:/Users/maria/OneDrive/Desktop/test_pc/2D/"


def plot2D(X, Y, mux, muy):
    radius = 3

    tl_x = min(np.min(X[:, 0]), np.min(Y[:, 0]), mux[0], muy[0]) - radius
    tl_y = min(np.min(X[:, 1]), np.min(Y[:, 1]), mux[1], muy[1]) - radius

    tr_x = max(np.max(X[:, 0]), np.max(Y[:, 0]), mux[0], muy[0]) + radius
    tr_y = max(np.max(X[:, 1]), np.max(Y[:, 1]), mux[1], muy[1]) + radius

    image_width = int(tr_x - tl_x)
    image_height = int(tr_y - tl_y)

    image = np.zeros((image_height, image_width))
    mu_x_pix = (int( (mux[0] - tl_x)), int( (mux[1] - tl_y)))
    cv2.circle(image, mu_x_pix, 7, color=(1), thickness=-1)

    mu_y_pix = (int( (muy[0] - tl_x)), int( (muy[1] - tl_y)))
    cv2.circle(image, mu_y_pix, 5, color=(2), thickness=-1)

    for i in range(X.shape[0]):
        X_i = (int( (X[i, 0] - tl_x)), int( (X[i, 1] - tl_y)))
        cv2.circle(image, X_i, 3, color=(1), thickness = -1)
    for j in range(Y.shape[0]):
        Y_i = (int((Y[j, 0] - tl_x)), int((Y[j, 1] - tl_y)))
        cv2.circle(image, Y_i, 3, color=(2), thickness = -1)

    plt.imshow(image)


class BiomechanicalCpd(pycpd.RigidRegistration):
    def __init__(self, X, Y, spring_indexes = [0], alpha = 0.0, sigma=None):
        self.spring_indexes = spring_indexes
        self.N_real = X.shape[0]  # length of X without the springs points

        additional_row = np.zeros((len(self.spring_indexes), 3))
        for i, spring_index in enumerate(self.spring_indexes):
            additional_row[i] = Y[i, :]

        Y = Y[0:-2, :]
        if len(self.spring_indexes) > 0:
            X = np.concatenate((X, additional_row), axis=0)
        super().__init__(X=X, Y=Y, sigma2=sigma)
        self.alpha = alpha
        self.s = 1

    def expectation(self):

        P = np.sum((self.X[None, :, :] - self.TY[:, None, :]) ** 2, axis=2)
        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        self.P = np.divide(P, den)

        if len(self.spring_indexes) > 0:
            self.P[:, -len(self.spring_indexes)::] = 0
            for i, spring_index in enumerate(self.spring_indexes):
                self.P[spring_index, self.N_real + i] = self.alpha

        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)

    def get_means(self):
        # evaluating where the point
        # target point cloud mean
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0),
                        self.Np)
        # source point cloud mean
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)
        return muX, muY

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # target point cloud mean
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0),
                        self.Np)
        # source point cloud mean
        muY = np.divide(
            np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.X_hat = self.X - np.tile(muX, (self.N, 1))
        # centered source point cloud
        Y_hat = self.Y - np.tile(muY, (self.M, 1))

        plot2D(self.X, self.TY, muX, muY)

        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
        # Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

        self.t = np.transpose(muX) - \
            np.dot(np.transpose(self.R), np.transpose(muY))

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()


def main(root_dir):

    source_img = cv2.imread(root_dir + "source.png", 0)
    target_img = cv2.imread(root_dir + "target.png", 0)

    plt.imshow(source_img)
    plt.show()

    source1 = np.argwhere(source_img == 141)
    source2 = np.argwhere(source_img == 131)
    target = np.argwhere(target_img == 111)
    springs_connections = np.mean(np.argwhere(source_img == 166), axis=0)
    springs_connections2 = np.mean(np.argwhere(source_img == 110), axis=0)

    source1 = np.concatenate((source1, np.zeros((source1.shape[0], 1))), axis=1)
    target = np.concatenate((target, np.zeros((target.shape[0], 1))), axis=1)

    random_idxes = np.random.randint(0, source1.shape[0], (1048))
    source1 = source1[random_idxes, ...]

    random_idxes = np.random.randint(0, target.shape[0], (1048*2))
    target = target[random_idxes, ...]

    source1[-2, 0:2] = springs_connections
    source1[-1, 0:2] = springs_connections2
    spring_indexes = [-2, -1]

    cpd_method = BiomechanicalCpd(target, source1, spring_indexes=[], alpha=1000, sigma=None)
    plt.ion()
    for iter in range(100):
        print("iter: " + str(iter))

        cpd_method.expectation()
        plt.show()
        plt.pause(1)
        cpd_method.maximization()


    #cpd_method.expectation()



main(root)