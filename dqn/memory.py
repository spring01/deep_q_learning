
import random
import cPickle as pickle

class ReplayMemory(object):

    def __init__(self, max_size, num_frames):
        self.maxlen = int(max_size / num_frames)
        self.num_frames = num_frames
        self.clear()

    def append(self, transition):
        self.ring_buffer[self.index] = transition
        self.index = (self.index + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def sample(self, batch_size):
        idx = random.sample(xrange(self.length), batch_size)
        return [self.ring_buffer[i] for i in idx]

    def clear(self):
        self.ring_buffer = [None for _ in xrange(self.maxlen)]
        self.index = 0
        self.length = 0

    def size(self):
        return self.length * self.num_frames

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)
