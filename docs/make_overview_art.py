"""Render the energy-landscape artwork used by docs/algorithm-overview.zh.html.

The surface is decorative: it illustrates a multi-well objective whose wells are
the near-optimal subsets a sampler returns. It is not a plot of any Q matrix.

    python3 docs/make_overview_art.py
"""

from __future__ import annotations

import io
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = pathlib.Path(__file__).resolve().parent / "assets"

PEAKS = (
    # x, y, height, width
    (-1.80, -0.30, 1.05, 0.42),
    (-0.70, 0.55, 0.70, 0.36),
    (0.35, -0.45, 1.45, 0.40),
    (1.45, 0.45, 0.95, 0.38),
    (2.45, -0.55, 0.60, 0.34),
    (-2.70, 0.50, 0.55, 0.36),
)

GOLD = ("#f6e2bd", "#f2c777", "#f0a63c", "#ef8b16", "#ffd27a")


def surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.zeros_like(x)
    for cx, cy, height, width in PEAKS:
        z += height * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2)))
    z += 0.06 * np.sin(1.7 * x) * np.cos(1.3 * y)
    return z


def _gold_colors(values: np.ndarray) -> np.ndarray:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gold", GOLD)
    lo, hi = float(values.min()), float(values.max())
    return cmap((values - lo) / max(hi - lo, 1e-9))


def render(path: pathlib.Path, seed: int = 7) -> pathlib.Path:
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(12.0, 6.4), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Dense cloud: points are kept with a probability that follows the surface,
    # so the wells read as bright clusters instead of a smooth sheet.
    n = 260_000
    x = rng.uniform(-3.2, 3.2, n)
    y = rng.uniform(-1.6, 1.6, n)
    z = surface(x, y)
    density = np.clip(z, 0.0, None) / z.max()
    keep = rng.random(n) < (0.030 + 0.70 * density**2.2)
    x, y, z = x[keep], y[keep], z[keep]
    jitter = rng.normal(0.0, 0.035, z.shape)
    ax.scatter(
        x,
        y,
        z + jitter,
        s=rng.uniform(0.25, 2.2, z.shape),
        c=_gold_colors(z),
        alpha=0.55,
        linewidths=0,
        depthshade=False,
    )

    # Faint floor texture under the wells.
    m = 9_000
    fx = rng.uniform(-3.4, 3.4, m)
    fy = rng.uniform(-1.8, 1.8, m)
    ax.scatter(
        fx,
        fy,
        np.full(m, 0.0),
        s=rng.uniform(0.2, 1.1, m),
        c="#d8b887",
        alpha=0.22,
        linewidths=0,
        depthshade=False,
    )

    # Light columns mark the deepest wells, i.e. the candidates worth keeping.
    for cx, cy, height, _ in sorted(PEAKS, key=lambda p: -p[2])[:3]:
        top = surface(np.array([cx]), np.array([cy]))[0]
        ax.plot(
            [cx, cx],
            [cy, cy],
            [top - 0.05, top + 0.70],
            color="#ffb43a",
            lw=1.6,
            alpha=0.55,
            solid_capstyle="round",
        )
        ax.scatter([cx], [cy], [top + 0.73], s=80, c="#ffd98a", alpha=0.9, linewidths=0)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_zlim(0.0, 2.6)
    ax.set_box_aspect((3.1, 1.3, 1.75))
    ax.view_init(elev=20, azim=-68)
    ax.set_axis_off()

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    # Trim the transparent margin so the page can size the figure predictably.
    buffer.seek(0)
    image = Image.open(buffer)
    box = image.getchannel("A").point(lambda v: 255 if v > 6 else 0).getbbox()
    image.crop(box).save(path)
    return path


if __name__ == "__main__":
    print(render(OUT_DIR / "energy-landscape.png"))
