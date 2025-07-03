from jsonargparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from lotka_volterra.model import Parameters, InitialConditions, vector_field, integrate

parser = ArgumentParser()
parser.add_argument("--parameters", type=Parameters)
parser.add_argument("--init_conds", type=InitialConditions)
parser.add_argument("--t1", type=float, default=100)

parser.add_argument("-c", "--config", action="config")

def main(config: dict | None = None) -> None:
    if config is None:
        config = parser.parse_args()

    config = parser.instantiate_classes(config)

    print(config)

    α, β, γ, δ = config.parameters.to_tuple()
    x0_y0 = config.init_conds.to_tuple()
    t1 = config.t1

    f = vector_field(α, β, γ, δ)

    solution = integrate(f, x0_y0, t1, dense_output=True, max_step=1e-2)

    print(solution)

    t = np.linspace(0, t1, 1000)
    x, y = solution.sol(t)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, x, label="x")
    ax1.plot(t, y, label="y")
    ax1.legend()
    ax2.plot(x, y)
    fig.savefig("result.png")


if __name__ == "__main__":
    main()
