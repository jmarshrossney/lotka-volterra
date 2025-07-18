from dataclasses import dataclass, field
from random import randrange
from typing import Callable

import equinox
import diffrax
from jsonargparse.typing import Path_fr
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from lotka_volterra.jax_model import (
    Parameters,
    InitialConditions,
    vector_field,
    VectorField,
    integrate,
)
from lotka_volterra.scripts.run_jax import parser

parser.add_argument("-d", "--data", type=Path_fr)

def evaluate_mse(solution: diffrax.Solution, data_xy: jax.Array, data_t: jax.Array) -> jax.Array:
    model_xy = jax.vmap(solution.evaluate)(data_t.squeeze()).transpose()
    return jnp.mean((model_xy - data_xy)**2) 

def loss_and_solution(
        parameters: Parameters,
        init_conds: InitialConditions,
        data_xy: jax.Array,
        data_t: jax.Array,
):
    solution = integrate(
        vector_field,
        init_conds.as_array(),
        t1=data_t.max() + 1e-3,
        dt0=0.1,
        args=parameters.as_tuple(),
        saveat=diffrax.SaveAt(dense=True),
    )
    loss = evaluate_mse(solution, data_xy, data_t)
    return (loss, solution)


def main():
    config = parser.parse_args()
    
    print("Config\n------\n" + parser.dump(config))

    # Instantiate `Parameters` and `InitialConditions` classes
    config = parser.instantiate_classes(config)

    data = np.loadtxt(config.data)
    data = jnp.asarray(data)
    data_t, data_xy = jnp.split(data, [1])
    print("data: ", data_t.shape, data_xy.shape)

    # Float dataclasses to jax dataclasses (this should not be necessary)
    parameters = Parameters(**config.parameters.to_dict_greek())
    init_conds = InitialConditions(*config.init_conds.to_tuple())

    loss, solution = loss_and_solution(parameters, init_conds, data_xy, data_t)

    grad_fn = jax.grad(loss_and_solution, argnums=0, has_aux=True)

    η = 0.001 # learning rate

    mse_series = []

    for i in range(1000):
        grad, _ = grad_fn(parameters, init_conds, data_xy, data_t)

        parameters = Parameters(
            α=parameters.α - η * grad.α,
            β=parameters.β - η * grad.β,
            γ=parameters.γ - η * grad.γ,
            δ=parameters.δ - η * grad.δ,
        )

        if i % 10 == 0:
            mse, _ = loss_and_solution(parameters, init_conds, data_xy, data_t)
            print(mse)
            mse_series.append(mse)

    _, solution = loss_and_solution(parameters, init_conds, data_xy, data_t)
    model_t = jnp.linspace(0, data_t.max(), 1000)
    model_xy = jax.vmap(solution.evaluate)(model_t).transpose()
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(data_t[0], data_xy[0], marker="x", linestyle="", label="data x")
    ax.plot(data_t[0], data_xy[1], marker="x", linestyle="", label="data y")
    ax.plot(model_t, model_xy[0], marker="", linestyle="-", label="model x")
    ax.plot(model_t, model_xy[1], marker="", linestyle="-", label="model y")
    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("x, y")
    fig.savefig("fit.png")

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(mse_series)) * 10, mse_series)
    ax.set_xlabel("training progress")
    ax.set_ylabel("MSE")
    fig.savefig("loss.png")



if __name__ == "__main__":

    with jax.default_device("cpu"): # cpu much faster in this case!
        main()
