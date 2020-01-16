import DPO.helper as helper
import numpy as np


def compute_lipschitz_constraints(container, intervals, sink, arriving_mcrst_helper, id_actions, theta, L):

    constraints = []
    # Lipschitz hypothesis between actions in the same macrostate.
    for cont in container:

        actions_mcrst = sorted(list(cont.keys()), reverse=True)
        new_mcrst_possible = []  # I collect all the possible arriving macrostate obtained from the same starting mcrst.
        for act in actions_mcrst:
            new_mcrst = helper.get_mcrst(cont[act]['new_state'], intervals, sink)

            if new_mcrst not in new_mcrst_possible:
                new_mcrst_possible.append(new_mcrst)

            # The helper might contain new_mcrst that are not yet included in new_mcrst_possible.
            from_helper = arriving_mcrst_helper[act].keys()
            for mcrst in from_helper:
                m = helper.get_mcrst_from_index(mcrst, intervals)
                if m not in new_mcrst_possible:
                    new_mcrst_possible.append(m)

        for d in range(len(intervals)):  # d represents the dimension.
            sep_mcrst = separate_mcrst(d, new_mcrst_possible)  # I divide the arriving mcrsts according to the
                                                                    # value of the mcrst on the dimension I'm evaluating.

            for i in range(0, len(actions_mcrst) - 1):  # i represents the index of the action in action_mcrst.
                action_index1 = id_actions[actions_mcrst[i]]
                action_index2 = id_actions[actions_mcrst[i + 1]]  # I obtain the id in the matrix of two adjacents actions.

                for k, mcrst_list in sep_mcrst.items():  # mcrst_list contain the mcrst to sum up
                    c1 = 0
                    c2 = 0
                    for m in mcrst_list:
                        c1 += theta[action_index1][helper.get_multidim_mcrst(m, intervals)]
                        c2 += theta[action_index2][helper.get_multidim_mcrst(m, intervals)]
                    constraints.append(c1 - c2 <= L[d][0] * abs(actions_mcrst[i] - actions_mcrst[i + 1]))
                    constraints.append(c1 - c2 >= -L[d][0] * abs(actions_mcrst[i] - actions_mcrst[i + 1]))

    return constraints


def separate_mcrst(d, mcrst):
    separator = {}
    for m in mcrst:
        if m[d] not in separator.keys():
            separator[m[d]] = [m]
        else:
            separator[m[d]].append(m)
    return separator
