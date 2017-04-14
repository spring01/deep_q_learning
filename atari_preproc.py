
import numpy as np
from PIL import Image


class AtariPreprocessor(object):

    def __init__(self, resize):
        self.resize = resize

    def frame_to_frame_mem(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize(self.resize)
        width = self.resize[0]
        height = self.resize[1]
        img = img.crop((0, height - width, width, height))
        return np.asarray(img)

    def state_mem_to_state(self, state_mem):
        state = np.stack(state_mem, axis=2)
        return state.astype(np.float32)

    def clip_reward(self, reward):
        if reward > 0.0:
            return 1.0;
        elif reward < 0.0:
            return -1.0
        else:
            return 0.0
