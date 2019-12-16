import numpy as np
import lqg1Dscalable.helper as helper


# given a pair (x, a) it returns an array representing min and max p(x'|x,a) for each x'
def abstract_tf(intervals, new_state_bounds, sink):

    adder = 1 if sink else 0
    abs_tf = []
    for i in range(0, len(intervals) + adder):
        abs_tf.append([0, 0])

    for ns in new_state_bounds:
        min_mcrst = helper.get_mcrst(ns[0], intervals, sink)
        max_mcrst = helper.get_mcrst(ns[1], intervals, sink)

        # update min interval.
        if min_mcrst == max_mcrst:
            abs_tf[min_mcrst][0] += 1

        # update max interval.
        for i in range(min_mcrst, max_mcrst + 1):
            abs_tf[i][1] += 1

        # correction (ev).
        if min_mcrst == -1 and max_mcrst == len(intervals):
            abs_tf[min_mcrst][1] -= 1

    # normalization.
    den = len(new_state_bounds)
    return [[el[0] / den, el[1] / den] for el in abs_tf]
    # return average_tf([[el[0] / den, el[1] / den] for el in abs_tf])


def average_tf(probabilities):
    arr = [(p[0] + p[1]) / 2 for p in probabilities]
    den = sum(arr)
    return [a/den for a in arr]
