from dataclasses import dataclass, field
from random import randrange
from typing import Callable

import equinox
from diffrax import diffeqsolve, Dopri5, ODETerm
import jax
import jax.numpy as jnp
import jax.tree_util


class Parameters(equinox.Module):
    """
    Parameters for the Lotka-Volterra model.

    Parameters:
      alpha: The base growth rate for prey.
      beta: Coefficient of predation.
      gamma: The base death rate for predators.
      delta: The influence of prey on the predator growth rate.
    """

    α: jax.Array
    β: jax.Array
    γ: jax.Array
    δ: jax.Array

    def as_tuple(self) -> tuple[jax.Array, ...]:
        return (self.α, self.β, self.γ, self.δ)


@dataclass
class InitialConditions:
    """
    Initial conditions for the Lotka-Volterra model.

    Parameters:
      x0: Initial population of prey.
      y0: Initial population of predators.
    """

    x0: jax.Array = field(default_factory=lambda: jnp.array(randrange(10, 20)))
    y0: jax.Array = field(default_factory=lambda: jnp.array(randrange(1, 8)))

    def as_array(self) -> jax.Array:
        return jnp.stack([jnp.asarray(self.x0), jnp.asarray(self.y0)])


type VectorField = Callable[[jax.Array, jax.Array], jax.Array]


def _vector_field(
    α: jax.Array, β: jax.Array, γ: jax.Array, δ: jax.Array
) -> VectorField:
    def f(t: jax.Array, x_y: jax.Array) -> jax.Array:
        x, y = x_y
        dxdt = α * x - β * x * y
        dydt = -γ * y + δ * x * y
        return jnp.stack([dxdt, dydt])

    return f


def vector_field(t: jax.Array, x_y: jax.Array, params: list[jax.Array]) -> jax.Array:
    return _vector_field(*params)(t, x_y)


def integrate(f: VectorField, x0_y0: jax.Array, t1: float, dt0: float, **kwargs):
    return diffeqsolve(
        terms=ODETerm(f),
        solver=Dopri5(),  # equivalent to RK45 in scipy
        t0=0,
        t1=t1,
        dt0=dt0,
        y0=x0_y0,
        **kwargs,
    )
