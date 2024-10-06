import numpy as np

from .ifunc import Relative


class NesterovFunc(Relative):

    def __init__(self, alpha: float = 0.1, d: int = 11, L: float = 1, k: int = 11):
        super().__init__(alpha, d)
        self.L = L
        self.k = k

        self.A = np.diag(np.concatenate((-np.ones(k - 1), [0] * (d - k))), k=-1) \
                 + np.diag(np.concatenate((-np.ones(k - 1), [0] * (d - k))), k=1) \
                 + 2 * np.diag(np.concatenate((np.ones(k), [0] * (d - k))), k=0)
    
    def __call__(self, x: np.ndarray) -> float:
        return self.L / 8 * x.T @ self.A @ x - self.L / 4 * x[0]

    def grad(self, x: np.ndarray) -> np.ndarray:
        e0 = np.zeros(self.d)
        e0[0] = 1
        return self.L / 4 * (self.A @ x -  e0)

    def x_star(self, x: np.ndarray) -> np.ndarray:
        return np.append(1 - np.arange(1, self.k + 1) / (self.d + 1), x[self.k:])


class SimpleFunc(Relative):

    def __init__(self, alpha: float = 0.1, L: float = 1):
        super().__init__(alpha, 2)
        self.L = L
        self.d = 2
        self.x_star = np.zeros(2)

    def __call__(self, x: np.ndarray) -> float:
        return self.L / 2 * x[0] ** 2

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.L * x[0], 0])


class SimpleMuFunc(Relative):

    def __init__(self, alpha: float = 0.1, _ = 2, L: float = 1, mu: float = 1):
        super().__init__(alpha, 2)
        self.L = L
        self.mu = mu
        self.d = 2

    def x_star(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def __call__(self, x: np.ndarray) -> float:
        return self.mu / 2 * x[0] ** 2 + self.L / 2 * x[1] ** 2

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.mu * x[0], self.L * x[1]])


class NesterovMUFunc(Relative):

    def __init__(self, alpha: float = 0.1, d: int = 11, L: float = 10, mu: float = 1):
        super().__init__(alpha, d)
        self._x_star = None
        self.L = L
        self.mu = mu
        self.A = np.diag(-np.ones(d - 1), k=-1) \
                 + np.diag(-np.ones(d - 1), k=1) \
                 + 2 * np.diag(np.ones(d), k=0)
        self.coef = self.L / self.mu

    def __call__(self, x: np.ndarray) -> float:
        return self.mu * (self.coef - 1) / 8 * (x.T @ self.A @ x  - 2 * x[0]) + self.mu / 2 * np.linalg.norm(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        e0 = np.zeros(self.d)
        e0[0] = 1
        return self.mu * (self.coef - 1) / 4 * (self.A @ x -  e0) + self.mu * x

    def x_star(self, _) -> np.ndarray:
        if self._x_star is not  None:
            return self._x_star
        e0 = np.zeros(self.d)
        e0[0] = 1
        l = self.mu * (self.coef - 1) / 4
        self._x_star = 2 * np.linalg.inv(self.A + self.mu / l * np.eye(self.d)) @ e0
        return self._x_star
