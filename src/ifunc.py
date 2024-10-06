from abc import abstractmethod, ABC
import numpy as np


class Relative(ABC):
    
    def __init__(self, alpha: float = 0.1, d: int = 11):
        self.alpha = alpha
        self.d = d

    def get_cos(self) -> float:
        return np.sqrt(1 - self.alpha ** 2)

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def estimation_grad(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def x_star(self, x: np.ndarray) -> np.ndarray:
        pass
