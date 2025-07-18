from datetime import datetime
import pathlib

from diffrax import SaveAt
import jax.numpy as jnp
from jsonargparse import ArgumentParser
from jsonargparse.typing import PositiveFloat

from lotka_volterra.model import (
    Parameters as FloatParameters,
    InitialConditions as FloatInitialConditions,
)
from lotka_volterra.jax_model import (
    Parameters,
    InitialConditions,
    vector_field,
    integrate,
)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


parser = ArgumentParser()

# Parameters required to specify the ODE problem to solve
parser.add_argument("--parameters", type=FloatParameters)
parser.add_argument("--init_conds", type=FloatInitialConditions)
parser.add_argument("--t1", type=PositiveFloat, help="Integration window size, [0, t1]")

parser.add_argument("-c", "--config", action="config")

# Draw default values from this configuration file
default_config_file = pathlib.Path(__file__).parent / "default_config_jax.yaml"
parser.default_config_files = [default_config_file]


def main(config: dict | None = None) -> None:
    """
    Integrates the Lotka-Volterra system and saves a plot.
    """
    if config is None:
        # Parse args from the command line
        config = parser.parse_args()

    print("Config\n------\n" + parser.dump(config))

    # Instantiate `Parameters` and `InitialConditions` classes
    config = parser.instantiate_classes(config)

    # Float dataclasses to jax dataclasses (this should not be necessary)
    parameters = Parameters(**config.parameters.to_dict_greek())
    init_conds = InitialConditions(*config.init_conds.to_tuple())

    # Unpack config object
    α, β, γ, δ = parameters.as_tuple()
    x0_y0 = init_conds.as_array()
    print(x0_y0)
    t1 = config.t1

    # f = vector_field(α, β, γ, δ)
    f = vector_field

    t = jnp.linspace(0, t1, 1000)
    solution = integrate(
        f, x0_y0, t1, dt0=0.1, args=parameters.as_tuple(), saveat=SaveAt(ts=t)
    )
    # x_y = solution.evaluate(t)
    print(solution.ys.shape)
    x_y = solution.ys.transpose()
    print(x_y.shape)

    import matplotlib.pyplot as plt

    x, y = x_y
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, x, label="x")
    ax1.plot(t, y, label="y")
    ax2.plot(x, y)
    ax1.legend()
    fig.savefig("jax_output.png")


if __name__ == "__main__":
    main()
