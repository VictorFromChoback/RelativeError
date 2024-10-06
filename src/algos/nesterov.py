import numpy as np

from ..ialgo import Algo
from ..ifunc import Relative

class Nesterov(Algo):

    def __init__(self, start, L, mu, func: Relative):
        super().__init__(start, L, mu, func)
        self.h = (((1 - self.func.alpha) / (1 + self.func.alpha)) ** (1.5)) / L
        self.corrected_L = ((1 + self.func.alpha) / ((1 - self.func.alpha) ** 3)) * L
        self.rho = 1 + self.func.alpha
        self.nu = 1 - self.func.alpha
        self.s = self.rho ** 2 + self.func.alpha ** 2
        self.m = self.nu ** 2 - self.func.alpha ** 2
        self.q = self.mu / self.corrected_L
        self.coef = self._calc_coef()
        self.u0 = start
        self.x0 = start
        self.u = [self.u0]
        self.x = [self.x0]
        self.y = []
        self.history.append(self.x0)

    def _calc_coef(self) -> float:
        return ((self.m - self.s) + np.sqrt((self.s - self.m) ** 2 + 4 * self.m * self.q)) / 2 / self.m

    def step(self):
        self.iter += 1  
        y = (self.x[-1] + self.coef * self.u[-1]) / (1 + self.coef)
        g = self.func.grad(y)
        u = (1 - self.coef) * self.u[-1] + self.coef * y - self.coef / self.mu * g
        x = y - self.h * g
        self.y.append(y)
        self.x.append(x)
        self.u.append(u)
        self.history.append(x)
