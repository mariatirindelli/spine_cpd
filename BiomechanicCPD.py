import numpy as np
import pycpd
from typing import Tuple, List
import utils
import matplotlib.pyplot as plt


class BiomechanicalCpd(pycpd.RigidRegistration):
    def __init__(self, X, Y, springs: List[Tuple] = [], alpha=0.0, sigma=None, max_iterations=None, R=None, t=None,
                 fix_variance=False):
        self.spring_indexes = [item[0] for item in springs]

        self.alpha = alpha
        self.N_real = X.shape[0]  # number of points without considering the one added to add the spring constraint
        self.fix_variance = fix_variance
        self.X = X

        if len(springs) > 0:
            self.add_springs_point_to_target(springs)

        # Initializing after adding springs or before?
        super().__init__(X=self.X, Y=Y, sigma2=sigma, max_iterations=max_iterations, R=R, t=t)
        self.s = 1  # setting the scaling to one as the transformation is fully rigid

        if len(springs) > 0:
            self.view_plane_x = springs[0][1][0, 0]
            self.view_plane_y = springs[0][1][0, 1]
            self.view_plane_z = springs[0][1][0, 2]
        else:
            self.view_plane_x = np.mean(self.X[:, 0])
            self.view_plane_y = np.mean(self.X[:, 1])
            self.view_plane_z = np.mean(self.X[:, 2])

    def get_view_plane(self, view_plane_position, view_plane):

        x_slice = utils.extract_pc_plane(self.X, view_plane_position, plane_ax=view_plane, plane_thickness = 5)
        ty_slice = utils.extract_pc_plane(self.TY, view_plane_position, plane_ax=view_plane, plane_thickness=5)

        translation, image_height, image_width = utils.get_image_properties(x_slice, ty_slice, radius=5)
        image = utils.initialize_image(image_height, image_width, 0.1)
        image = utils.add_pc2image(x_slice, image, translation, 0.1, color=1, radius=10)
        image = utils.add_pc2image(ty_slice, image, translation, 0.1, color=2, radius=10)
        return image

    def save_reg_results(self, **kwargs):

        image_x = self.get_view_plane(self.view_plane_x, 0)
        image_y = self.get_view_plane(self.view_plane_y, 1)
        image_z = self.get_view_plane(self.view_plane_z, 2)

        for i, image in enumerate([image_x, image_y, image_z]):
            plt.subplot(1, 3, i+1)
            plt.imshow(image)
        plt.show()

    def add_springs_point_to_target(self, springs):
        additional_row = np.zeros((len(springs), 3))
        self.X = np.concatenate((self.X, additional_row), axis=0)
        self.update_springs(springs)

    def update_springs(self, springs):

        assert len(self.spring_indexes) == len(springs)

        self.spring_indexes = [item[0] for item in springs]
        additional_points = [item[1] for item in springs]
        self.X[-len(self.spring_indexes)::, :] = 0

        for i, _ in enumerate(self.spring_indexes):
            self.X[-len(self.spring_indexes)+i, :] = additional_points[i]

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

        if np.max(np.abs(self.P)) == 0:
            self.P = self.P + np.finfo(float).eps

        if len(self.spring_indexes) > 0:
            self.update_spring_probabilities()

        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()

        if not self.fix_variance:
            self.update_variance()

    def update_spring_probabilities(self):
        self.P[:, -len(self.spring_indexes)::] = 0
        for i, spring_index in enumerate(self.spring_indexes):
            self.P[spring_index, self.N_real + i] = self.alpha

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
