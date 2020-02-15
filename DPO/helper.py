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


def count_actions(container):
    n_actions = 0
    for i in range(0, len(container)):
        n_actions += len(list(container[i].keys()))
    return n_actions


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
    for act in cont.keys():
        rew += np.dot(cont[act]['state'], np.dot(Q, cont[act]['state'])) + np.dot(action, np.dot(R, action))
    rew = rew.item()
    return - rew / len(cont.items())


# def calc_abs_reward_cartpole(cont, action):
#     return 1 if -0.2093 <= cont[action]['state'] <= 0.2093 else 0


def calc_abs_reward_minigolf(cont, action):
    action = np.clip(action, 0, 5)
    rew = 0
    for act in cont.keys():
        if action < np.sqrt(1.836 * cont[act]['state'][0]):
            rew += -1
        elif action > np.sqrt(7.33 + 1.836 * cont[act]['state'][0]):
            rew += -100
        else:
            rew += 0
    return rew / len(cont.items())


# .................................................


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
    zeros = 0
    hundred = 0
    # max_action = 0
    failing_states = []
    for sam in samples:
        for s in sam:
            # if s[1] > max_action:
            #     max_action = s[1]
            if s[2] == 0:
                zeros += 1
            if s[2] == -100:
                hundred += 1
                failing_states.append(s[0])

    # print("Max action: {}".format(max_action))
    return zeros, hundred, failing_states


def interval_intersection(bounds):
    mins = [b[0] for b in bounds]
    maxs = [b[1] for b in bounds]
    max_mins = mins[0]
    min_maxs = maxs[0]
    for m in mins:
        max_mins = np.maximum(max_mins, m)  # I obtain the max value of mins for each dimension
    for m in maxs:
        min_maxs = np.minimum(min_maxs, m)  # I obtain the min value of maxs for each dimension
    if np.all(np.minimum(max_mins, min_maxs) == max_mins):
        return max_mins, min_maxs
    else:
        # return None, None  # void intersection
        clip_s = manage_clip_error(bounds)
        return clip_s, clip_s


def build_mcrst_from_samples(samples, n_mcrst, min_val, max_val):
    samples_base = flat_listoflists(samples)
    samples_base = [s[0] for s in samples_base]
    INTERVALS = []

    for i in range(len(n_mcrst)):
        samples = sorted([s[i] for s in samples_base])
        mcrst_size = math.floor(len(samples) / n_mcrst[i])
        dim_int = []

        while len(samples) >= 2 * mcrst_size:
            if len(dim_int) == 0:
                dim_int.append([min_val[i], samples[mcrst_size - 1]])
            else:
                dim_int.append([dim_int[-1][1], samples[mcrst_size - 1]])
            samples = samples[mcrst_size:]
        if len(samples) > 0:
            if len(dim_int) == 0:
                dim_int.append([min_val[i], max_val[i]])
            else:
                dim_int.append([dim_int[-1][1], max_val[i]])
        INTERVALS.append(dim_int)

    return INTERVALS


# Used to calculate new state with a Lipschitz hypothesis on delta s.
# It works only with the bounds obtained with the double integrator.
def manage_clip_error(bounds):
    if ishorizontal(bounds[0]):
        news = [bounds[1][0][0], bounds[0][0][1]]
    elif ishorizontal(bounds[1]):
        news = [bounds[0][0][0], bounds[1][0][1]]
    elif isvertical(bounds[0]):
        news = [bounds[0][0][0], bounds[1][0][1]]
    elif isvertical(bounds[1]):
        news = [bounds[1][0][0], bounds[0][0][1]]
    else:
        news = bounds[0][0]
    return np.array(news)

    # b = flat_listoflists(bounds)
    # coord = b[0].reshape((-1, 1))
    # for c in b[1:]:
    #     coord = np.append(coord, c.reshape((-1, 1)), axis=1)
    # sol = [mode(c)[0][0] for c in coord]
    # return np.array(sol), np.array(sol)


def ishorizontal(bound):
    return (bound[0][0] != bound[1][0]) and (bound[0][1] == bound[1][1])


def isvertical(bound):
    return (bound[0][0] == bound[1][0]) and (bound[0][1] != bound[1][1])


def ispoint(bound):
    return (bound[0][0] == bound[1][0]) and (bound[0][1] == bound[1][1])

# test
# interv = [[[], []], [[], [], []], [[], [], [], []]]
# mul_mcrst = [1, 1, 3]
# print(get_multidim_mcrst(mul_mcrst, interv))


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


def sq_distance(arr1, arr2):
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

def offset_prop(angle, offset):
    if angle < 180:
        angle -= angle / offset
    elif angle > 180:
        angle += (angle - 360) * (-1) / offset
    return angle

def offset_sum(angle, offset):
    if 0 < angle < 90 or 180 < angle < 270:
        angle += offset
    elif 90 < angle < 180 or 270 < angle < 360:
        angle -= offset
    return angle

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


def appendix_mcrst_population(samples, mask, intervals):
    intervals = np.array([i for i, m in zip(intervals, mask) if m])
    shape = np.array([len(i) for i in intervals])
    cont = np.zeros(tuple(shape))
    for s in samples:
        s = s[0]
        s = np.array([s_ for s_, m in zip(s, mask) if m])
        mcrst = get_mcrst(s, intervals, False)
        cont[tuple(mcrst)] += 1
    return cont
