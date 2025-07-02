# Lotka-Volterra

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
