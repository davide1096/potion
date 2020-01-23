import DPO.helper as helper
import numpy as np


def compute_lipschitz_constraints(intervals, ordered_actions, arriving_mcrst, theta, L):

    # Lipschitz hypothesis between actions in the same macrostate.
    constraints = []

    for d in range(len(intervals)):  # d represents the dimension.
        sep_mcrst = separate_mcrst(d, arriving_mcrst, intervals)  # I divide the arriving mcrsts on each dimension.

        for i in range(0, len(ordered_actions) - 1):  # i represents the index of the action in action_mcrst.

            for k, mcrst_list in sep_mcrst.items():  # mcrst_list contain the mcrst to sum up
                c1 = 0
                c2 = 0
                for m in mcrst_list:
                    c1 += theta[i][helper.get_multidim_mcrst(m, intervals)]
                    c2 += theta[i+1][helper.get_multidim_mcrst(m, intervals)]
                constraints.append(c1 - c2 <= L[d][0] * abs(ordered_actions[i] - ordered_actions[i + 1]))
                constraints.append(c1 - c2 >= -L[d][0] * abs(ordered_actions[i] - ordered_actions[i + 1]))

    return constraints


def separate_mcrst(d, arriving_mcrst, intervals):
    separator = {}
    for m in arriving_mcrst:
        m = helper.get_mcrst_from_index(m, intervals)
        if m[d] not in separator.keys():
            separator[m[d]] = [m]
        else:
            separator[m[d]].append(m)
    return separator
