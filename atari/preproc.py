
from dqn.preprocessor import Preprocessor


class AtariPreprocessor(Preprocessor):

    def __init__(self, resize):
        self.resize = resize
        self.width = self.height = self.resize[0]
        resize_ht = self.resize[1]
        self.crop_tuple = 0, resize_ht - self.width, self.width, resize_ht

