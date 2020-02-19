import random
import numpy as np
import math
from scipy.stats import mode

MAX_SAMPLES_IN_MCRST = 60


class Helper(object):

    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        random.seed(self.seed)

    def big_mcrst_correction(self, cont):
        new_cont = {}
        for i in range(0, MAX_SAMPLES_IN_MCRST):
            rdm = random.randint(0, len(cont.keys()) - 1)
            index = list(cont.keys())[rdm]
            new_cont[index] = cont[index]
        return new_cont

    def getSeed(self):
        return self.seed


def get_mcrst(state, intervals, sink):

    if not hasattr(state, "__len__"):
        state = [state]

    mcrst = []
    for dim in range(len(intervals)):
        dim_int = intervals[dim]

        if sink and state[dim] < dim_int[0][0]:
            mcrst.append(-1)
        elif sink and state[dim] > dim_int[-1][1]:
            mcrst.append(len(dim_int))

        elif state[dim] >= dim_int[-1][1]:   # above the upper bound of the state space
            mcrst.append(len(dim_int) - 1)
        elif state[dim] <= dim_int[0][0]:    # below the lower bound of the state space
            mcrst.append(0)

        else:
            index = 0                       # inside the state space
            for inter in dim_int:
                if inter[0] <= state[dim] < inter[1]:
                    mcrst.append(index)
                    break
                else:
                    index = index + 1
    return mcrst


# --> helper function to compute the index in container related to the exact macrostate <---

# it returns the index from a multidim mcrst.
def get_index_from_mcrst(multi_mcrst, intervals):
    mcrst = 0
    for i in range(len(multi_mcrst)):
        if multi_mcrst[i] == -1 or multi_mcrst[i] == len(intervals[i]):
            return 'sink'
        mcrst += multi_mcrst[len(multi_mcrst) - i - 1] * product_prev_sizes(i, intervals)
    return mcrst


def product_prev_sizes(i, intervals):
    prod = 1
    for d in range(i):
        prod *= len(intervals[len(intervals) - 1 - d])
    return prod


# given an index return the related macrostate expressed with a value for each dimension.
def get_mcrst_from_index(index, intervals):
    mcrst = []
    for i in range(len(intervals)):
        prod = product_prev_sizes(len(intervals) - 1 - i, intervals)
        ind = math.floor(index / prod)
        mcrst.append(ind)
        index -= prod*ind
    assert(index == 0)
    return mcrst

# -----------------------------------------------------------------------------------------

def normalize_array(array):
    den = np.sum(array)
    return array/den


def normalize_dict(dic):
    den = 0
    for k, v in dic.items():
        den += v
    for k in dic.keys():
        dic[k] = dic[k]/den
    return dic


def flat_listoflists(list):
    return [item for sublist in list for item in sublist]


# function used in abstract reward calculation


def calc_abs_reward_lqg(cont, action, Q, R, maxa_env):
    rew = 0
    action = np.clip(action, -maxa_env, maxa_env)
    for id in cont.keys():
        rew += np.dot(cont[id]['state'], np.dot(Q, cont[id]['state'])) + np.dot(action, np.dot(R, action))
    rew = rew.item()
    return - rew / len(cont.items())


def estimate_J_from_samples(samples, gamma):
    acc = 0
    for sam in samples:
        g = 1
        for s in sam:
            # sum of discounted rewards of the initial samples.
            acc += g * s[2]
            g *= gamma
    return acc / len(samples)


def minigolf_reward_counter(samples):
    hundred = 0
    failing_states = []
    for s in samples:
        if s[2] == -100:
            hundred += 1
            failing_states.append(s[0])
    return hundred, failing_states


def get_constant_intervals(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN):
    mcrst = []
    for mins, maxs, n in zip(MIN_SPACE_VAL, MAX_SPACE_VAL, N_MCRST_DYN):  # for every dimension
        dim = math.floor(((maxs - mins) / n) * 100)/100.0  # diameter of the macrostate
        remaining = round(((maxs - mins) - dim * n), 2)  # correction
        intervals = []
        counter = round(mins + dim + math.floor(remaining/2 * 100)/100.0, 2)
        intervals.append([mins, counter])
        while counter <= round(maxs - 2 * dim, 2):
            intervals.append([counter, round(counter + dim, 2)])
            counter = round(counter + dim, 2)
        if counter < maxs:
            intervals.append([counter, maxs])
        mcrst.append(intervals)
    return mcrst


def array_in(arr, arr_list):
    for a in arr_list:
        if np.all(arr == a):
            return True
    return False


def arr_distance(arr1, arr2):
    if not hasattr(arr1, "__len__"):
        return abs(arr1-arr2)
    else:
        d = 0
        for d1, d2 in zip(arr1, arr2):
            d += (d1 - d2) ** 2
        return d

# given cos and sin return the angle in degrees
def get_angle(x, y):
    alpha_rad = np.arctan(y/x)
    alpha = 57.3 * alpha_rad
    if alpha > 0 and x < 0:  # correction arctan
        alpha += 180
    if alpha < 0 and x < 0:  # correction arctan
        alpha += 180
    if alpha < 0:  # correction neg alpha
        alpha += 360
    return alpha

def offset_sum2(angle, offset):
    angle += offset
    if angle > 360:
        angle -= 360
    return angle

def get_sin_cos(angle):
    angle = angle / 57.3
    return math.sin(angle), math.cos(angle)

def bias_compass_observation(x, y, offset):
    alpha = get_angle(x, y)
    alpha = offset_sum2(alpha, offset)
    sin_alpha, cos_alpha = get_sin_cos(alpha)
    return cos_alpha, sin_alpha
