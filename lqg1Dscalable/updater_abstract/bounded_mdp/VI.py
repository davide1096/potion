import numpy as np


def update(bounded_v, ordered_v, action, cont, gamma):

    rew = cont[action]['abs_reward']
    bound_tf = cont[action]['abs_tf']

    new_tf = [b[0] for b in bound_tf]
    remaining = 1 - sum(new_tf)

    for el in ordered_v:
        # I get the upper bound of the tf probability related to a state ordered among the first ones.
        ub = bound_tf[el][1]
        diff = ub - new_tf[el]

        # I compute the TF related to the MDP that I'm searching for.
        if diff <= remaining:
            new_tf[el] = ub
        else:
            new_tf[el] += remaining

        remaining = max(0, remaining - diff)

    return rew + gamma * np.dot(new_tf, bounded_v)
