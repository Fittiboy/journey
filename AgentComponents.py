import numpy as np


class ParameterSchedule:
    def __init__(self, schedule_fn, start, end):
        # schedule_fn(0) should be 0, and not exceed 1
        self.fn = schedule_fn
        self.start = start
        self.end = end
        assert self.fn(0) == 0
        xs = np.linspace(0, 1)
        vals = self.fn(xs)
        assert np.max(vals) <= 1

    def value(self, ep, max_eps):
        w = self.fn(ep / max_eps)
        return (1 - w) * self.start + w * self.end


class LinearSchedule(ParameterSchedule):
    def __init__(self, start, end):
        super().__init__(linear_weight, start, end)


class SigmoidSchedule(ParameterSchedule):
    def __init__(self, start, end):
        super().__init__(sigmoid_weight, start, end)


class UpDownSchedule(ParameterSchedule):
    def __init__(self, start, end):
        super().__init__(updown_weight, start, end)


def linear_weight(p):
    return p


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_weight(p):
    return (sigmoid(p*6) - 0.5) / (sigmoid(6) - 0.5)


def updown_weight(p):
    return (- (p - 0.5) ** 2) * 4 + 1