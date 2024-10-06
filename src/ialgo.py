from abc import abstractmethod, ABC

from .ifunc import Relative


class Algo(ABC):

    def __init__(self, start, L, mu, func: Relative):
        self.L = L
        self.mu = mu
        self.func = func
        self.start = start
        self.y0 = start
        self.func = func
        self.iter = 0
        self.history = []

    @abstractmethod
    def step(self):
        # change state
        pass
