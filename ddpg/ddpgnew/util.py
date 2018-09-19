import numpy as np


class BoxSpaceNormalizer():
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.half = (high - low) / 2

    def norm(self, real_value):
        return (real_value - self.low - self.half) / self.half

    def denorm(self, normed_value):
        return normed_value * self.half + self.half + self.low
