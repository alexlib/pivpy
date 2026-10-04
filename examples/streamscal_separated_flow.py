"""Two-panel streamscal figure of a synthetic separated flow.

The field is analytic, not a measurement and not a CFD result: a tilted
free-stream shear layer (softplus stream function) plus three Gaussian
vortices that form recirculation zones next to the left and bottom walls.
`sep` scales the vortex strength, so the two panels show a strong and a weak
separation. Both panels share one colour scale (scalar="u").

    python examples/streamscal_separated_flow.py
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np

import pivpy.pivpy  # noqa: F401  (registers the .piv accessor)
from pivpy import graphics
from pivpy.schema import build_dataset


def separated_flow(nx=64, ny=72, sep=1.0):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.15, ny)
    X, Y = np.meshgrid(x, y)

    # free stream confined above a diagonal shear layer
    psi = 0.22 * np.logaddexp(0.0, (Y + 0.55 * X - 0.85) / 0.22) * 0.9
    # (cx, cy, width, signed strength) of the three recirculation vortices
    for cx, cy, s, g in [(0.16, 0.95, 0.11, -0.075),
                         (0.14, 0.38, 0.10, 0.075),
                         (0.52, 0.07, 0.13, -0.075)]:
        psi += sep * g * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s**2))

    u = np.gradient(psi, y, axis=0)
    v = -np.gradient(psi, x, axis=1)
    return build_dataset(x, y, u, v)


if __name__ == "__main__":
    strong, weak = separated_flow(sep=1.0), separated_flow(sep=0.3)
    fig, axs = graphics.streamscal_panels([strong, weak], figwidth=12, scalar="u")
    fig.savefig("streamscal_separated_flow.png", dpi=110)
