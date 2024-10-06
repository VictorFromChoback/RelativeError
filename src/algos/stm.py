import numpy as np

from ..ialgo import Algo
from ..ifunc import Relative

class STM(Algo):

    def __init__(self, start, L, mu, func: Relative):
        super().__init__(start, L, mu, func)
        self.a0 = 1 / L
        self.A0 = 1 / L
        self.cura = self.a0
        self.curA = self.A0
        self.y0 = start
        self.h = 1
        self.z0 = start - self.h / L * func.estimation_grad(start, start, func.x_star(start))
        self.x0 = self.z0
        self.y = [self.y0]
        self.z = [self.z0]
        self.x = [self.x0]
        self.history.append(self.x0)

    def step(self):
        self.iter += 1
        self.preva = self.cura
        self.prevA = self.curA
        t = (1 + self.mu * self.prevA)
        self.cura = t / 2 / self.L +\
                    + np.sqrt(t ** 2 / 4 / self.L / self.L
                              + self.prevA * t / self.L)
        self.curA = self.prevA + self.cura
        y = self.prevA / self.curA * self.x[-1] + self.cura / self.curA * self.z[-1]
        z = self.z[-1] - self.h * self.cura / (1 + self.mu * self.curA) * (
            self.func.grad(y) + self.mu * (self.z[-1] - y)
        )
        
        x = self.prevA / self.curA * self.x[-1] + self.cura / self.curA * z
        z_noised = self.z[-1] - self.h * self.cura / (1 + self.mu * self.curA) * (
            self.func.estimation_grad(y, x, self.func.x_star(x)) + self.mu * (self.z[-1] - y)
        )
        x_noised = self.prevA / self.curA * self.x[-1] + self.cura / self.curA * z_noised
        self.z.append(z_noised)
        x = self.prevA / self.curA * self.x[-1] + self.cura / self.curA * z_noised
        self.y.append(y)
        self.x.append(x_noised)
        self.history.append(x_noised)
