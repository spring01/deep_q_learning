
import time
import numpy as np
from PIL import Image


class Preprocessor(object):

    def __init__(self, resize):
        self.resize = resize
        self.width, self.height = self.resize
        self.crop_tuple = 0, 0, self.width, self.height

    def frame_to_frame_mem(self, state):
        img = Image.fromarray(state)
        img = img.convert('L')
        img = img.resize(self.resize)
        img = img.crop(self.crop_tuple)
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

    def show_effect(self, state):
        img_original = Image.fromarray(state)
        img_processed = Image.fromarray(self.frame_to_frame_mem(state))
        img_original.show()
        time.sleep(0.1)
        img_processed.show()


class BottomSquarePreprocessor(Preprocessor):

    def __init__(self, resize):
        self.resize = resize
        self.width = self.height = self.resize[0]
        resize_ht = self.resize[1]
        self.crop_tuple = 0, resize_ht - self.width, self.width, resize_ht


class TopSquarePreprocessor(Preprocessor):

    def __init__(self, resize):
        self.resize = resize
        self.width = self.height = self.resize[0]
        self.crop_tuple = 0, 0, self.width, self.width


