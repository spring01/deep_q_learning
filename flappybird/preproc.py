
from dqn.preprocessor import Preprocessor


class FlappyBirdPreprocessor(Preprocessor):

    def __init__(self, resize):
        self.resize = resize
        self.width = self.resize[0]
        self.height = int(self.width * 1.414)
        self.crop_tuple = (0, 0, self.width, self.height)

