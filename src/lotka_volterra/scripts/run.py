from datetime import datetime
import pathlib

from jsonargparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from lotka_volterra.model import Parameters, InitialConditions, vector_field, integrate

def filename() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{timestamp}.png"


parser = ArgumentParser()
parser.add_argument("--parameters", type=Parameters)
parser.add_argument("--init_conds", type=InitialConditions)
parser.add_argument("--t1", type=float, default=100, help="Integration window size, [0, t1]")
parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path(filename()), help="file path to save plot.")

parser.add_argument("-c", "--config", action="config")

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

    f = vector_field(α, β, γ, δ)

    result = integrate(f, x0_y0, t1, dense_output=True, max_step=1e-2)
    
    # `result` contains a bunch of info. We just want the solution.
    solution = result.sol

    t = np.linspace(0, t1, 1000)
    x, y = solution(t)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, x, label="x")
    ax1.plot(t, y, label="y")
    ax2.plot(x, y)
    ax1.legend()
    fig.savefig(config.output)


if __name__ == "__main__":
    main()
