from dataclasses import dataclass, field, fields
from random import randrange
from typing import Callable

from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult  # for type annotation only


@dataclass
class Parameters:
    """
    Parameters for the Lotka-Volterra model.

    Parameters:
      alpha: The base growth rate for prey.
      beta: Coefficient of predation.
      gamma: The base death rate for predators.
      delta: The influence of prey on the predator growth rate.
    """

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
    """
    Initial conditions for the Lotka-Volterra model.

    Parameters:
      x0: Initial population of prey.
      y0: Initial population of predators.
    """

    x0: float = field(default_factory=lambda: randrange(10, 20))
    y0: float = field(default_factory=lambda: randrange(1, 8))

    def __post_init__(self) -> None:
        if any([getattr(self, field.name) < 0 for field in fields(self)]):
            raise ValueError("Initial populations should be non-negative!")

    def to_tuple(self) -> tuple[float, float]:
        return (self.x0, self.y0)


type VectorField = Callable[[float, tuple[float, float]], tuple[float, float]]


def vector_field(α: float, β: float, γ: float, δ: float) -> VectorField:
    """
    Constructs a vector field on the RHS of the Lotka-Volterra equations.

        d(x, y)/dt = f(t, (x, y))

    where

        dx/dt = αx - βxy ; dy/dt = -γy + δxy

    Note that there is no time (t) dependence.
    """

    def f(t: float, x_y: tuple[float, float]) -> tuple[float, float]:
        x, y = x_y
        dxdt = α * x - β * x * y
        dydt = -γ * y + δ * x * y
        return (dxdt, dydt)

    return f


def integrate(
    f: VectorField, x0_y0: tuple[float, float], t1: float, **kwargs
) -> OdeResult:
    """
    A thin wrapper for scipy.integrate.solve_ivp that integrates a vector field.

    Parameters:
      f: A vector field with signature f(t, (x, y))
      x0_y0: A tuple containing the initial (t=0) states for x & y
      t1: The time to integrate up to
      kwargs: Additional arguments to pass to `solve_ivp`.
    """
    return solve_ivp(
        fun=f,
        t_span=(0, t1),
        y0=x0_y0,
        **kwargs,
    )
