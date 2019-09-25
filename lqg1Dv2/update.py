INIT_V_PARAM = 0.1
LR_STOCH_POLICY = 0.01
LR_VFUN = 0.1


class Updater(object):

    def __init__(self, n_mcrst, gamma, n_steps):
        super().__init__()
        self.v_params = [INIT_V_PARAM for i in range(0, n_mcrst)]
        self.gamma = gamma
        # For n_step I need the value related to the abstract sampling
        self.n_steps = n_steps

    def policy_gradient_update(self, samples, policy):
        d_factor = 1
        index = 0
        policy = clean_policy(policy, samples)
        for s in samples:
            delta = s[2] + self.gamma * self.v_params[s[3]] - self.v_params[s[0]]
            # update v_params
            self.v_params[s[0]] += LR_VFUN * delta
            # update abstract policy parameters
            policy[s[0]] = update_mcrst_policy(policy[s[0]], s[1], LR_STOCH_POLICY * d_factor * delta)
            # during each episode the discount factor needs to be updated
            d_factor = d_factor * self.gamma if index < (self.n_steps - 1) else 1
            index = index + 1 if index < (self.n_steps - 1) else 0
        return policy


def is_action_sampled(action, samples):
    for s in samples:
        if s[1] == action:
            return True
    return False


# remove from the policy the actions that are not in the abstract samples:
# their parameter won't be updated and this can decrease the effect of the updates
def clean_policy(policy, samples):
    for i in range(0, len(policy)):
        policy[i] = list(filter(lambda x: is_action_sampled(x[0], samples), policy[i]))
    return [normalize_prob_array(p) for p in policy]


# policy is the policy related to a specific macrostate
def update_mcrst_policy(policy, action, partial_prod):
    for p in range(0, len(policy)):
        if policy[p][0] == action:
            update = partial_prod / policy[p][1] if not policy[p][1] == 0 else 0
            policy[p][1] += update
    # policy = list(filter(lambda x: x[1] > 0, policy))
    # to avoid probabilities < 0
    minor = min(p[1] for p in policy)
    if minor < 0:
        for i in range(0, len(policy)):
            policy[i][1] += minor
    return normalize_prob_array(policy)


def normalize_prob_array(policy):
    den = 0
    for p in range(0, len(policy)):
        den += policy[p][1]
    for p in range(0, len(policy)):
        policy[p][1] = policy[p][1] / den
    return policy
