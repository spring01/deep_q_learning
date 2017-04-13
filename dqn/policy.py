
import numpy as np


class UniformRandomPolicy(object):

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self):
        return np.random.randint(0, self.num_actions)


class GreedyPolicy(object):

    def select_action(self, q_values, *args):
        return np.argmax(q_values)


class GreedyEpsilonPolicy(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, q_values, *args):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, q_values.size)
        else:
            return np.argmax(q_values)


class LinearDecayGreedyEpsilonPolicy(object):

    def __init__(self, start_value, end_value, num_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = float(num_steps)

    def select_action(self, q_values, iter_num=0):
        wt_end = min(iter_num / self.num_steps, 1.0)
        wt_start = 1.0 - wt_end
        epsilon = self.start_value * wt_start + self.end_value * wt_end
        if np.random.rand() <= epsilon:
            return np.random.randint(0, q_values.size)
        else:
            return np.argmax(q_values)
