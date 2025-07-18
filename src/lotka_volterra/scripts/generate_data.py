from datetime import datetime
import pathlib

from jsonargparse import ArgumentParser
from jsonargparse.typing import PositiveInt, PositiveFloat
import matplotlib.pyplot as plt
import numpy as np

from lotka_volterra.model import Parameters, InitialConditions, vector_field, integrate


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


parser = ArgumentParser()

# Parameters required to specify the ODE problem to solve
parser.add_argument("--parameters", type=Parameters)
parser.add_argument("--init_conds", type=InitialConditions)
parser.add_argument("--t1", type=PositiveFloat, help="Integration window size, [0, t1]")

# Parameters related to the generated dataset
parser.add_argument("--n_points", type=PositiveInt, help="Number of points to sample")
parser.add_argument(
    "--sigma_x", type=PositiveFloat, help="Standard deviation of Gaussian noise (prey)"
)
parser.add_argument(
    "--sigma_y",
    type=PositiveFloat,
    help="Standard deviation of Gaussian noise (predators)",
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=timestamp(),
    help="name for the output files (without a suffix)",
)

parser.add_argument("-c", "--config", action="config")

# Draw default values from this configuration file
default_config_file = pathlib.Path(__file__).parent / "default_config.yaml"
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

    # Unpack config object
    α, β, γ, δ = config.parameters.to_tuple()
    x0_y0 = config.init_conds.to_tuple()
    t1 = config.t1
    n_points = config.n_points
    σ_x, σ_y = config.sigma_x, config.sigma_y
    file_stem = config.output

    f = vector_field(α, β, γ, δ)

    result = integrate(f, x0_y0, t1, dense_output=True, max_step=1e-2)

    # `result` contains a bunch of info. We just want the solution.
    solution = result.sol

    # Initialise random number generator
    rng = np.random.default_rng()

    # Sample time points uniformly within the interval, and sort them
    t = np.sort(rng.uniform(0, t1, size=[n_points]))

    # Obtain deterministic solution at these time points
    x_y = solution(t)

    # Add Gaussian noise to x, y
    δx_δy = rng.normal(loc=0, scale=(σ_x, σ_y), size=[n_points, 2]).transpose()
    x_y += δx_δy

    # Save output to .txt
    t_x_y = np.concatenate([np.expand_dims(t, 0), x_y])
    np.savetxt(f"{file_stem}.txt", t_x_y, fmt="%.4e")

    # Plot data
    x, y = x_y
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, x, label="x")
    ax1.plot(t, y, label="y")
    ax2.plot(x, y)
    ax1.legend()
    fig.savefig(f"{file_stem}.png")


if __name__ == "__main__":
    main()
