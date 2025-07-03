from dataclasses import dataclass, field, fields
from os import PathLike
from random import randrange
from typing import Any, Callable, Self

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

@dataclass
class Parameters:
    alpha: float
    beta: float
    gamma: float
    delta: float

    def __post_init__(self) -> None:
        if any([getattr(self, field.name) < 0 for field in fields(self)]):
            raise ValueError("All parameters should be non-negative!")

    def to_tuple(self) -> tuple[float, ...]:
        return (self.alpha, self.beta, self.gamma, self.delta)

    def to_dict_greek(self) -> dict[str, float]:
        return {"α": self.alpha, "β": self.beta, "γ": self.gamma, "δ": self.delta}

@dataclass
class InitialConditions:
    x0: float = field(default_factory=lambda: randrange(10, 20))
    y0: float = field(default_factory=lambda: randrange(1, 8))
    
    def __post_init__(self) -> None:
        if any([getattr(self, field.name) < 0 for field in fields(self)]):
            raise ValueError("Initial populations should be non-negative!")

    def to_tuple(self) -> tuple[float, float]:
        return (self.x0, self.y0)

type VectorField = Callable[[float, tuple[float, float]], tuple[float, float]]

def vector_field(α: float, β: float, γ: float, δ: float) -> VectorField:
    def f(t: float, x_y: tuple[float, float]) -> tuple[float, float]:
        x, y = x_y
        dxdt = α * x - β * x * y
        dydt = -γ * y + δ * x * y
        return (dxdt, dydt)
    return f


def integrate(f: VectorField, x0_y0: tuple[float, float], t1: float, **kwargs) -> OdeResult:
    return solve_ivp(
        fun=f,
        t_span=(0, t1),
        y0=x0_y0,
        **kwargs,
    )
