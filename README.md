# Lotka-Volterra

WIP + a bit of fun/learning for me.

## Background

The basic Lotka-Volterra model is described by the ODE system

$$
\begin{aligned}
\frac{\mathrm{d} x}{\mathrm{d} t} &= \alpha x - \beta x y \\
\frac{\mathrm{d} y}{\mathrm{d} t} &= -\gamma y + \delta x y
\end{aligned}
$$

where $x(t), y(t), \alpha, \beta, \gamma, \delta > 0$.

In the 'predator-prey' interpretation of the model,

- $x(t)$, $y(t)$ are time ($t$) dependent population densities for 'prey' and 'predator' species,
- $\alpha$ is the prey growth rate,
- $\beta$ encodes the effect of predators on the prey's growth rate,
- $\gamma$ is the predator death rate.
- $\delta$ encodes the effect of prey on the predators growth rate.

## Developer setup

1. Clone this repository

```sh
git clone https://github.com/jmarshrossney/lotka-volterra
cd lotka-volterra
```

2. Install `uv` by following [these instructions](https://docs.astral.sh/uv/#installation)

3. Run the following to create a virtual environment and install all the packages (including this one)

```sh
uv sync
```

4. Check that it works by running the example

```sh
uv run example
```


## Developer tools

To lint and format the package, run

```sh
uv run ruff check
```

in the root of the repository.

If you have configured pre-commit hooks, this will be run automatically upon `git commit`.
