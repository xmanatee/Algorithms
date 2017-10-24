import numpy as np


class SharpeRatio:
    def __init__(self, std_precision=1e-6):
        self.std_precision = std_precision
        self.num = None
        self.sum = None
        self.sqr_sum = None
        self.reset()

    def reset(self):
        self.num = 0
        self.sum = 0
        self.sqr_sum = 0

    def add(self, value):
        self.num += 1
        self.sum += value
        self.sqr_sum += value ** 2

    def get(self):
        assert self.num != 0, "Need at least one number."
        mean = self.sum / self.num
        sqr_mean = self.sqr_sum / self.num
        s2 = sqr_mean - mean ** 2

        std = np.sqrt(s2) + self.std_precision
        return mean / std
