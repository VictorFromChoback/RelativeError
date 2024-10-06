import numpy as np

from ..ialgo import Algo
from ..ifunc import Relative

class HeavyBall(Algo):

    def __init__(self, start, L, mu, func: Relative):
        super().__init__(start, L, mu, func)
        self.h = 4 / ((np.sqrt(L) + np.sqrt(mu)) ** 2)
        self.beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
        x = start - self.h * func.estimation_grad(start, start, func.x_star(start))
        self.history.append(start)
        self.history.append(x)

    def step(self):
        self.iter += 1
        prev = self.history[-1]
        prevprev = self.history[-2]
        x = prev - self.h * self.func.grad(prev) + self.beta * (prev - prevprev)
        xnoised = prev - self.h * self.func.estimation_grad(prev, x, self.func.x_star(x)) + self.beta * (prev - prevprev)
        self.history.append(xnoised)
