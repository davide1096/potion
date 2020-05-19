import sys
sys.path.append("/Users/davide/Desktop/Uni/Thesis/Code/davide1096:potion/potion")
import numpy as np
from DPO.helper import flat_listoflists
from scipy import stats
# import matplotlib.pyplot as plt
test_samples = [[[3.982295, 3.229995, -1.0, -1.232816], [-1.232816, 0, 0.0,
                                                                                      -1.2328167268212766]],
                [[6.138535, 3.813946, -1.0, -0.822523], [-0.822523, 0, 0.0, -0.822523]], [[13.4820315, 4.069156, -1.0, 6.563384], [6.563384, 3.886516, -1.0, -0.604846]]]


def ls_deltas_weights(samples):
    # samples: is in the format given by sampling_from_det_pol(), i.e. list of trajectories, where each trajectory is
    #          a list of elements [s,a,r,s']

    sam = np.array(flat_listoflists(samples))
    # print(sam)
    target = sam[:, 3] - sam[:, 0]
    input_states = sam[:, 0]
    input_actions = sam[:, 1]
    a = np.column_stack((np.ones(target.size), input_states, input_actions, np.multiply(input_states, input_actions),
                         np.power(input_states, 2), np.power(input_actions, 2)))
    weights = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(target)
    # print(a)

    return weights


def ls_pred(point, weights):
    ar = [point[0], point[1], point[0]*point[1], point[0]*point[0], point[1]*point[1]]
    x_ = np.array([1] + ar)
    y_ = x_.dot(weights)
    return y_

# print(ls_deltas_weights(test_samples))
# print(ls_pred([1.747175193440356, 1.1872242835138265], ls_deltas_weights(test_samples)))

class Ridge(object):
    def __init__(self, alpha=1.0, learn_rate=0.01, decay=0):
        self.alpha = alpha
        self.x_means = None
        self.y_mean = None
        self.std_dev = None
        self.n_samples = None
        self.learn_rate = learn_rate
        self.avg_error = 0
        self.std_weights = None
        self.decay = decay

    def ls_fit(self, X, T):
        # Input variables get standardized, while output variable just gets centered into 0
        self.x_means = np.array([np.mean(k) for k in X])
        self.y_mean = np.mean(T)
        self.std_dev = np.array([np.std(k) for k in X])
        std_x = np.array(np.divide(np.subtract(X, self.x_means.reshape((len(X), 1))), self.std_dev.reshape((len(X),1))))
        centered_y = np.array(np.subtract(T, self.y_mean))
        self.n_samples = len(X[0])

        self.std_weights = np.array(np.linalg.inv(std_x.dot(std_x.T) + self.alpha * np.ones((len(X),len(X)))).dot(
            std_x).dot(centered_y))

    def get_weights(self):
        # Returns the unstandardized weights
        w0 = np.array(self.y_mean - np.sum(np.divide(np.multiply(self.x_means, self.std_weights), self.std_dev)))
        w_ = np.array(np.divide(self.std_weights, self.std_dev))
        return np.append(w0, w_)

    def stats_update(self, X, T):
        new_n_samples = len(X[0]) + self.n_samples
        new_std_dev = np.array([np.std(k) for k in X])
        new_x_means = np.array(
            [np.mean(k) for k in X])
        self.std_dev = np.sqrt((self.n_samples/new_n_samples)*np.power(self.std_dev, 2.0) +
                               (len(X[0]/new_n_samples)*np.power(new_std_dev, 2.0)) +
                               ((self.n_samples*len(X[0]))/pow(new_n_samples, 2.0))*np.power(np.subtract(
            self.x_means, new_x_means), 2))
        self.x_means = (self.n_samples / new_n_samples) * self.x_means + (len(X[0]) / new_n_samples) * new_x_means
        self.y_mean = (self.n_samples / new_n_samples) * self.y_mean + (len(X[0]) / new_n_samples) * np.mean(T)
        self.n_samples = new_n_samples

    def mb_update(self, X, T, it):
        self.compute_error(X, T)
        self.x_means = np.array([np.mean(k) for k in X])
        self.y_mean = np.mean(T)
        self.std_dev = np.array([np.std(k) for k in X])
        centered_y = np.array(np.subtract(T, self.y_mean))
        m = len(X[0]) # number of samples
        std_x = np.array(np.divide(np.subtract(X, self.x_means.reshape((len(X), 1))), self.std_dev.reshape((len(X),1))))

        n_epochs = 200
        for i in range(n_epochs):
            pred_y = std_x.T.dot(self.std_weights)
            error = np.subtract(centered_y, pred_y)
            avg_grads = std_x.dot(error) / m

            if self.decay:
                n = it
            else:
                n = 1
            self.std_weights = self.std_weights - (self.learn_rate / n) * (-avg_grads + self.alpha * self.std_weights)

    def pred(self, X):
        w = self.get_weights()
        return (np.insert(X, 0, 1)).dot(w)

    def get_error(self):
        return self.avg_error

    def compute_error(self, X, T):
        new_error = np.abs(np.subtract(T, np.row_stack((np.ones(len(X[0])), X)).T.dot(self.get_weights())))
        self.avg_error = (np.average(new_error))



