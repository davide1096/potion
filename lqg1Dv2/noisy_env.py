from scipy.stats import norm


class NoisyEnvironment(object):

    def __init__(self, intervals):
        super().__init__()
        self.intervals = intervals

    # the sum of probabilities in general is lower than one.
    def get_mcrst_prob(self, mu, sigma):
        lim_ints = [i[0] for i in self.intervals]
        lim_ints.append(self.intervals[-1][1])
        lim_distr = norm(mu, sigma).cdf(lim_ints)
        return [lim_distr[i+1] - lim_distr[i] for i in range(0, len(self.intervals))]


# test = NoisyEnvironment([[-2, -1.6], [-1.6, -1.2], [-1.2, -1], [-1, -0.8], [-0.8, -0.6], [-0.6, -0.5], [-0.5, -0.4],
#                          [-0.4, -0.3], [-0.3, -0.2], [-0.2, -0.1], [-0.1, 0.], [0., 0.1], [0.1, 0.2], [0.2, 0.3],
#                          [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.8], [0.8, 1], [1, 1.2], [1.2, 1.6], [1.6, 2]])

