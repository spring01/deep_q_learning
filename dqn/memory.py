
import numpy as np
import cPickle as pickle


class PriorityMemory(object):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--memory_steps', default=100000, type=int,
            help='Replay memory size')
        parser.add_argument('--burn_in_steps', default=10000, type=int,
            help='Fill the replay memory to how much size before update')
        parser.add_argument('--mem_alpha', default=0.6, type=float,
            help='Exponent alpha in prioritized replay memory')
        parser.add_argument('--mem_beta_init', default=0.4, type=float,
            help='Initial beta in prioritized replay memory')

    def __init__(self, args, act_steps, train_steps):
        self.burn_in_steps = args.burn_in_steps
        self.alpha = args.mem_alpha
        self.beta_init = args.mem_beta_init
        self.train_steps = float(train_steps)
        self.maxlen = int(args.memory_steps / act_steps)
        self.indices = range(self.maxlen)
        self.act_steps = act_steps
        self.clear()

    def append(self, transition):
        self.ring_buffer[self.index] = transition
        self.priority[self.index] = np.max(self.priority)
        self.index = (self.index + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def sample(self, batch_size):
        prob = self.priority / np.sum(self.priority)
        batch_idx = np.random.choice(self.indices, batch_size, False, prob)
        batch = [self.ring_buffer[i] for i in batch_idx]
        batch_prob = prob[batch_idx]
        return batch, batch_idx, batch_prob

    def get_batch_weights(self, batch_idx, batch_prob, iter_num):
        wt_end = min(iter_num / self.train_steps, 1.0)
        wt_start = 1.0 - wt_end
        beta_annealed = self.beta_init * wt_start + 1.0 * wt_end
        batch_weights = (self.length * batch_prob)**(-beta_annealed)
        batch_weights /= np.max(batch_weights)
        return batch_weights

    def update_priority(self, batch_idx, batch_td_error):
        batch_priority = np.abs(batch_td_error)
        batch_priority[batch_priority > 1.0] = 1.0
        batch_priority[batch_priority == 0.0] = 1e-8
        self.priority[batch_idx] = batch_priority**self.alpha

    def clear(self):
        self.ring_buffer = [None for _ in xrange(self.maxlen)]
        self.priority = np.zeros(self.maxlen)
        self.priority[0] = 1.0
        self.index = 0
        self.length = 0

    def size(self):
        return self.length * self.act_steps

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        with open(filepath, 'rb') as save:
            memory = pickle.load(save)
        self.alpha = memory.alpha
        self.beta_init = memory.beta_init
        self.train_steps = memory.train_steps
        self.maxlen = memory.maxlen
        self.indices = memory.indices
        self.act_steps = memory.act_steps
        self.ring_buffer = memory.ring_buffer
        self.priority = memory.priority
        self.index = memory.index
        self.length = memory.length

