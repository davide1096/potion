import numpy as np
import DPO.helper as helper


def abstract_tf(intervals, new_states, sink):

    abs_tf = {}

    for ns in new_states:
        ns_mcrst = helper.get_mcrst(ns, intervals, sink)
        index_mcrst = helper.get_index_from_mcrst(ns_mcrst, intervals)
        if index_mcrst not in abs_tf:
            abs_tf[index_mcrst] = 1
        else:
            abs_tf[index_mcrst] += 1

    return helper.normalize_dict(abs_tf)
