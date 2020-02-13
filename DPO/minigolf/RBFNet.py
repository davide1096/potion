import numpy as np
import matplotlib.pyplot as plt
import random

# BATCH_SIZE = 50
# LAMBDA = 0.001


def rbf(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, centers, sigma, w, seed, lr=0.01, lam=0.001, k=5, epochs=200, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.lam = lam
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.centers = centers
        self.stds = np.repeat(sigma, k)
        self.w = w

        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def fit(self, X, y):

        old_w = self.w.copy()
        new_w = self.w.copy()
        to_shuffle = [[x, y_] for x, y_ in zip(X, y)]
        Xy = random.sample(to_shuffle, len(to_shuffle))
        X = [xy[0] for xy in Xy]
        y = [xy[1] for xy in Xy]

        for x, y_ in zip(X, y):
            x = x[0]
            # forward pass
            a = np.array([self.rbf(x, c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(new_w)

            # backward pass
            error = -(y_ - F).flatten()

            # online update
            grad = a * error

            reg = np.sign([neww - oldw for neww, oldw in zip(new_w, old_w)])  # L1-norm
            # reg = np.array([neww - oldw for neww, oldw in zip(new_w, old_w)])  # L2-norm
            new_w = new_w - self.lr * (grad + self.lam * reg)

        self.w = new_w

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = np.array([a.T.dot(self.w)])
            y_pred.append(F)
        return np.array(y_pred)
