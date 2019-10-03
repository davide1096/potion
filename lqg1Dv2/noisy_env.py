from scipy.stats import norm
from scipy.special import ndtr


class NoisyEnvironment(object):

    def __init__(self, intervals):
        super().__init__()
        self.intervals = intervals

    # the sum of probabilities in general is lower than one.
    def get_mcrst_prob(self, mu, sigma):
        min_int = [i[0] for i in self.intervals]
        mins_distr = norm(mu, sigma).cdf(min_int)
        maxs_distr = [mins_distr[i] for i in range(1, len(mins_distr))]
        maxs_distr.append(norm(mu, sigma).cdf(self.intervals[-1][1]))
        return [mx - mn for mn, mx in zip(mins_distr, maxs_distr)]


test = NoisyEnvironment([[-2, -1.6], [-1.6, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6], [-0.6, -0.5], [-0.5, -0.4],
                         [-0.4, -0.3], [-0.3, -0.2], [-0.2, -0.1], [-0.1, 0.], [0., 0.1], [0.1, 0.2], [0.2, 0.3],
                         [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.8], [0.8, 1], [1, 1.2], [1.2, 1.6], [1.6, 2]])

