import numpy as np


def update(bounded_v, ordered_v, action, cont, gamma):

    rew = cont[action]['abs_reward']
    bound_tf = cont[action]['abs_tf']
    split = np.split(bound_tf, 2, axis=0)

    new_tf = split[0][0].copy()
    remaining = 1 - np.sum(new_tf)

    for el in ordered_v:
        if remaining == 0:
            break

        if el == 'sink':
            el = [-1]

        # I get the upper bound of the tf probability related to a state ordered among the REINFORCE ones.
        ub = split[1][0][tuple(el)]
        diff = ub - new_tf[tuple(el)]

        # I compute the TF related to the MDP that I'm searching for.
        if diff <= remaining:
            new_tf[tuple(el)] = ub
        else:
            new_tf[tuple(el)] += remaining

        remaining = max(0, remaining - diff)

    return rew + gamma * np.sum(new_tf * bounded_v)
