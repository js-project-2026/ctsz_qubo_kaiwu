"""Render the figures used by docs/algorithm-overview.zh.html.

Two kinds of output land in docs/assets/:

* Decorative art (landscapes, the selection lattice, the coupling graph). The
  surfaces illustrate a multi-well objective; they are not plots of any Q.
* Charts, which carry real numbers from runs recorded in this repository.
  Every value is listed in CHART_SOURCES below next to the run it came from.

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

GOLD = "#e8930c"
GOLD_SOFT = "#f6b94a"
GOLD_PALE = "#f7dcae"
GREY = "#9a948a"
GREY_PALE = "#d7d2c7"
INK = "#16151a"
BAD = "#b3503a"

GOLD_RAMP = ("#f6e2bd", "#f2c777", "#f0a63c", "#ef8b16", "#ffd27a")

CHART_SOURCES = {
    # microglia GSE198531, 15,156 genes after filtering, alpha* = 0.3662
    "solvers": "microglia2.log: tabu 50 genes -1.2534, GA 51 genes -1.2478, MQLib 27 genes -1.0177",
    "mse": "microglia2.log downstream regression: all 0.0253, RF 0.0220, tabu 0.0210, GA 0.0203, LASSO 0.0177",
    # DPP-IV published split, 133 + 133 independent test, threshold 0.50
    "dppiv": "dppiv_narrow.log: ours 0.891/0.872/0.910/0.783/0.950; StackDPPIV Table 3 0.891/0.857/0.925/0.784/0.961",
    # fetal haematopoietic run, 4,463 cells, 2,500 genes, alpha* = 0.3203
    "cim": "benchmark_cim14: raw +9.8033 (315 genes), 1-bit refine -1.5862 (48 genes), Ocean -1.5888 (50 genes)",
}

PEAKS = (
    (-1.80, -0.30, 1.05, 0.42),
    (-0.70, 0.55, 0.70, 0.36),
    (0.35, -0.45, 1.45, 0.40),
    (1.45, 0.45, 0.95, 0.38),
    (2.45, -0.55, 0.60, 0.34),
    (-2.70, 0.50, 0.55, 0.36),
)


def surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.zeros_like(x)
    for cx, cy, height, width in PEAKS:
        z += height * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2)))
    z += 0.06 * np.sin(1.7 * x) * np.cos(1.3 * y)
    return z


def _gold_colors(values: np.ndarray) -> np.ndarray:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gold", GOLD_RAMP)
    lo, hi = float(values.min()), float(values.max())
    return cmap((values - lo) / max(hi - lo, 1e-9))


def _save(fig, path: pathlib.Path, *, crop: bool = True) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer)
    if crop:
        box = image.getchannel("A").point(lambda v: 255 if v > 6 else 0).getbbox()
        image = image.crop(box)
    image.save(path)
    return path


def landscape(path: pathlib.Path, *, seed: int, azim: float, elev: float,
              beams: int = 3, samples: int = 0) -> pathlib.Path:
    """Particle view of a multi-well objective; optional sampled solutions."""
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(12.0, 6.4), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    n = 260_000
    x = rng.uniform(-3.2, 3.2, n)
    y = rng.uniform(-1.6, 1.6, n)
    z = surface(x, y)
    density = np.clip(z, 0.0, None) / z.max()
    keep = rng.random(n) < (0.030 + 0.70 * density**2.2)
    x, y, z = x[keep], y[keep], z[keep]
    ax.scatter(
        x,
        y,
        z + rng.normal(0.0, 0.035, z.shape),
        s=rng.uniform(0.25, 2.2, z.shape),
        c=_gold_colors(z),
        alpha=0.55,
        linewidths=0,
        depthshade=False,
    )

    m = 9_000
    ax.scatter(
        rng.uniform(-3.4, 3.4, m),
        rng.uniform(-1.8, 1.8, m),
        np.zeros(m),
        s=rng.uniform(0.2, 1.1, m),
        c="#d8b887",
        alpha=0.22,
        linewidths=0,
        depthshade=False,
    )

    ranked = sorted(PEAKS, key=lambda p: -p[2])
    for cx, cy, _, _ in ranked[:beams]:
        top = surface(np.array([cx]), np.array([cy]))[0]
        ax.plot([cx, cx], [cy, cy], [top - 0.05, top + 0.70],
                color="#ffb43a", lw=1.6, alpha=0.55, solid_capstyle="round")
        ax.scatter([cx], [cy], [top + 0.73], s=80, c="#ffd98a", alpha=0.9, linewidths=0)

    # Sampled solutions: several wells carry an acceptable answer, so a sampler
    # returns a set of candidates rather than one point.
    for cx, cy, _, width in ranked[:samples]:
        jitter = rng.normal(0.0, width * 0.55, (14, 2))
        sx, sy = cx + jitter[:, 0], cy + jitter[:, 1]
        sz = surface(sx, sy) + 0.10
        ax.scatter(sx, sy, sz, s=26, c="#ffffff", edgecolors=GOLD,
                   linewidths=1.0, alpha=0.95, depthshade=False)

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_zlim(0.0, 2.6)
    ax.set_box_aspect((3.1, 1.3, 1.75))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return _save(fig, path)


def selection_lattice(path: pathlib.Path, seed: int = 3) -> pathlib.Path:
    """x in {0,1}^n drawn as a candidate pool with a few entries switched on."""
    rng = np.random.default_rng(seed)
    cols, rows = 34, 13
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    gx = gx.astype(float) + rng.normal(0, 0.06, gx.shape)
    gy = gy.astype(float) + rng.normal(0, 0.06, gy.shape)

    chosen = np.zeros(gx.size, dtype=bool)
    chosen[rng.choice(gx.size, 21, replace=False)] = True

    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.scatter(gx.ravel()[~chosen], gy.ravel()[~chosen], s=16, c=GREY_PALE, linewidths=0)
    ax.scatter(gx.ravel()[chosen], gy.ravel()[chosen], s=86, c=GOLD, linewidths=0, zorder=3)
    ax.scatter(gx.ravel()[chosen], gy.ravel()[chosen], s=250, c=GOLD, alpha=0.16,
               linewidths=0, zorder=2)

    # capacity ruler underneath
    y0 = -2.6
    ax.plot([0, cols - 1], [y0, y0], lw=7, color=GREY_PALE, solid_capstyle="round")
    ax.plot([0, (cols - 1) * 0.72], [y0, y0], lw=7, color=GOLD_SOFT, solid_capstyle="round")
    ax.plot([(cols - 1) * 0.72] * 2, [y0 - 0.9, y0 + 0.9], lw=2, color=GOLD)

    ax.set_xlim(-1.2, cols)
    ax.set_ylim(y0 - 2.0, rows)
    ax.set_axis_off()
    return _save(fig, path)


def coupling_graph(path: pathlib.Path, seed: int = 11) -> pathlib.Path:
    """R_ij drawn as a weighted graph: candidates that repeat each other."""
    rng = np.random.default_rng(seed)
    n = 150
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.sqrt(rng.random(n))
    px = radius * np.cos(angle) * 2.4
    py = radius * np.sin(angle)

    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for i in range(n):
        d = np.hypot(px - px[i], py - py[i])
        for j in np.argsort(d)[1:4]:
            if j <= i:
                continue
            w = float(np.exp(-d[j] * 2.0))
            ax.plot([px[i], px[j]], [py[i], py[j]], lw=0.6 + 1.6 * w,
                    color=GOLD_SOFT, alpha=0.12 + 0.45 * w, zorder=1)

    hub = rng.choice(n, 9, replace=False)
    ax.scatter(px, py, s=12, c=GOLD_PALE, linewidths=0, zorder=2)
    ax.scatter(px[hub], py[hub], s=92, c="#ffffff", edgecolors=GOLD, linewidths=1.6, zorder=3)
    ax.set_axis_off()
    return _save(fig, path)


def _bar_axes(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY_PALE)
    ax.spines["bottom"].set_color(GREY_PALE)
    ax.tick_params(colors="#6b6760", labelsize=8.5, length=3)
    return fig, ax


def chart_solvers(path: pathlib.Path) -> pathlib.Path:
    labels = ["Tabu\n50 genes", "Memetic GA\n51 genes", "MQLib 120 s\n27 genes"]
    values = [1.2534, 1.2478, 1.0177]
    colors = [GOLD, GOLD_SOFT, GREY]
    fig, ax = _bar_axes(5.3, 2.5)
    bars = ax.bar(labels, values, color=colors, width=0.56)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_ylim(0, 1.45)
    ax.set_ylabel("|E| at alpha* = 0.3662", fontsize=8.5, color="#6b6760")
    ax.grid(axis="y", color=GREY_PALE, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    return _save(fig, path, crop=False)


def chart_mse(path: pathlib.Path) -> pathlib.Path:
    labels = ["All 15,156\ngenes", "Random forest", "QUBO tabu\n50", "QUBO GA\n51", "LASSO\n48 nonzero"]
    values = [0.0253, 0.0220, 0.0210, 0.0203, 0.0177]
    colors = [GREY_PALE, GREY, GOLD, GOLD_SOFT, "#5b7fb4"]
    fig, ax = _bar_axes(5.3, 2.5)
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0006, f"{value:.4f}",
                ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_ylim(0, 0.030)
    ax.set_ylabel("held-out MSE (lower is better)", fontsize=8.5, color="#6b6760")
    ax.grid(axis="y", color=GREY_PALE, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    return _save(fig, path, crop=False)


def chart_dppiv(path: pathlib.Path) -> pathlib.Path:
    metrics = ["ACC", "Sn", "Sp", "MCC", "AUC"]
    ours = [0.891, 0.872, 0.910, 0.783, 0.950]
    paper = [0.891, 0.857, 0.925, 0.784, 0.961]
    idx = np.arange(len(metrics))
    fig, ax = _bar_axes(6.0, 2.6)
    b1 = ax.bar(idx - 0.19, ours, width=0.36, color=GOLD, label="ours, single model")
    b2 = ax.bar(idx + 0.19, paper, width=0.36, color=GREY, label="StackDPPIV, 50-model stack")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7.6, color=INK)
    ax.set_xticks(idx, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("independent test, 133 + 133", fontsize=8.5, color="#6b6760")
    ax.grid(axis="y", color=GREY_PALE, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.33))
    return _save(fig, path, crop=False)


def chart_cim(path: pathlib.Path) -> pathlib.Path:
    labels = ["CIM raw\n14-bit, 315 genes", "CIM + 1-bit refine\n48 genes", "Ocean tabu\n50 genes"]
    values = [9.8033, -1.5862, -1.5888]
    colors = [BAD, GOLD, GREY]
    fig, ax = _bar_axes(5.6, 2.5)
    bars = ax.bar(labels, values, color=colors, width=0.56)
    for bar, value in zip(bars, values):
        offset = 0.5 if value > 0 else -1.1
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.4f}",
                ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0, color="#6b6760", lw=0.9)
    ax.set_ylim(-3.2, 11.5)
    ax.set_ylabel("energy on the original float Q", fontsize=8.5, color="#6b6760")
    ax.grid(axis="y", color=GREY_PALE, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    return _save(fig, path, crop=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = [
        landscape(OUT_DIR / "energy-landscape.png", seed=7, azim=-68, elev=20),
        landscape(OUT_DIR / "landscape-sampled.png", seed=21, azim=-48, elev=26, beams=0, samples=4),
        landscape(OUT_DIR / "landscape-wide.png", seed=5, azim=-84, elev=13, beams=2),
        selection_lattice(OUT_DIR / "selection-lattice.png"),
        coupling_graph(OUT_DIR / "coupling-graph.png"),
        chart_solvers(OUT_DIR / "chart-solvers.png"),
        chart_mse(OUT_DIR / "chart-mse.png"),
        chart_dppiv(OUT_DIR / "chart-dppiv.png"),
        chart_cim(OUT_DIR / "chart-cim.png"),
    ]
    for item in made:
        print(item)


if __name__ == "__main__":
    main()
