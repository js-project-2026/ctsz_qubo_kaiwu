"""How much wet-lab work one screening pass replaces, stated as a formula.

Unit prices differ per vendor, per scale and per contract, so none are hard
coded here. What is fixed are the measured quantities: the classifier's
sensitivity and specificity on the published independent split, the panel size
against the transcriptome it was chosen from, and the size of the combination
space. Pass your own prices to turn those into money.

    python3 docs/cost_model.py
    python3 docs/cost_model.py --peptide-cost 1200 --hits 10
    python3 docs/cost_model.py --samples 400 --wts-cost 1800 --panel-cost 260

It also renders docs/assets/chart-enrichment.png for the deck.
"""

from __future__ import annotations

import argparse
import pathlib
from math import comb

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# dppiv_narrow.log, independent test set of 133 + 133, threshold fixed at 0.50
SN = 0.872
SP = 0.910

# microglia2.log: 15,156 genes after filtering, panel of 50
GENES_TOTAL = 15_156
PANEL_SIZE = 50
MSE_PANEL = 0.0210
MSE_ALL = 0.0253

# 原始多肽数据与预测结果.xlsx, sheet 1 detected peptides, sheet 2 shortlist
PEPTIDES_TOTAL = 69_415
PEPTIDES_PICKED = 25

# RGC_Repair_Gene_Atlas: genes that passed curation
RGC_GENES = 456

PRIORS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

OUT = pathlib.Path(__file__).resolve().parent / "assets" / "chart-enrichment.png"
GOLD = "#e8930c"
GREY = "#9a948a"
GREY_PALE = "#d7d2c7"
INK = "#16151a"


def ppv(prior: float, sn: float = SN, sp: float = SP) -> float:
    """Share of true actives among the peptides the model asks you to make."""
    tp = sn * prior
    fp = (1.0 - sp) * (1.0 - prior)
    return tp / (tp + fp)


def peptide_table(sn: float = SN, sp: float = SP) -> list[tuple[float, float, float, float]]:
    rows = []
    for prior in PRIORS:
        hit_rate = ppv(prior, sn, sp)
        rows.append((prior, hit_rate, hit_rate / prior, 1.0 / hit_rate))
    return rows


def report(args: argparse.Namespace) -> None:
    print("1. Peptide synthesis and IC50 assay")
    print(f"   library {PEPTIDES_TOTAL:,} detected peptides -> {PEPTIDES_PICKED} to synthesise "
          f"(1 in {PEPTIDES_TOTAL / PEPTIDES_PICKED:,.0f})")
    print(f"   measured Sn = {args.sn:.3f}, Sp = {args.sp:.3f}; a false positive costs one synthesis plus one assay")
    print("   syntheses needed per confirmed active = 1 / hit rate")
    print()
    print("   prior   hit rate   enrichment   syntheses per hit   without the model")
    for prior, hit_rate, factor, per_hit in peptide_table(args.sn, args.sp):
        print(f"   {prior:5.1%}   {hit_rate:8.1%}   {factor:9.1f}x   {per_hit:17.1f}   {1 / prior:17.0f}")
    print()
    if args.peptide_cost is not None:
        print(f"   at {args.peptide_cost:,.0f} per peptide (synthesis + assay), for {args.hits} confirmed actives:")
        for prior, hit_rate, _, per_hit in peptide_table(args.sn, args.sp):
            saved = args.hits * args.peptide_cost * (1.0 / prior - per_hit)
            print(f"   prior {prior:5.1%}: {args.hits * per_hit:6.0f} syntheses instead of "
                  f"{args.hits / prior:7.0f}, saving {saved:,.0f}")
    else:
        print("   saving = hits x unit cost x (1 / prior - 1 / hit rate); pass --peptide-cost to price it")
    print("   the prior for this library is not known yet; it is fixed by the first IC50 round")
    print()

    print("2. Targeted panel instead of whole transcriptome")
    print(f"   {PANEL_SIZE} of {GENES_TOTAL:,} genes = {PANEL_SIZE / GENES_TOTAL:.2%} of the transcriptome")
    print(f"   held-out MSE {MSE_PANEL:.4f} with the panel against {MSE_ALL:.4f} with every gene, "
          f"so the cut costs no accuracy")
    if args.samples and args.wts_cost is not None and args.panel_cost is not None:
        saving = args.samples * (args.wts_cost - args.panel_cost)
        print(f"   {args.samples} samples at {args.wts_cost:,.0f} against {args.panel_cost:,.0f}: "
              f"saving {saving:,.0f}")
    else:
        print("   saving = samples x (whole-transcriptome price - panel price); pass --samples, "
              "--wts-cost and --panel-cost to price it")
    print()

    print("3. Multi-gene combinations under one vector")
    for k in (2, 3, 4):
        print(f"   {RGC_GENES} curated genes, pick {k}: {comb(RGC_GENES, k):,} combinations")
    print("   one animal cohort tests one construct, so the question is never which is best overall,")
    print("   but which few constructs are worth building; the capacity term removes the rest before synthesis")


def chart(path: pathlib.Path = OUT, sn: float = SN, sp: float = SP) -> pathlib.Path:
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
    plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
    plt.rcParams["axes.unicode_minus"] = False

    priors = np.logspace(np.log10(0.003), np.log10(0.30), 200)
    per_hit_model = np.array([1.0 / ppv(p, sn, sp) for p in priors])
    per_hit_blind = 1.0 / priors

    fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY_PALE)
    ax.spines["bottom"].set_color(GREY_PALE)
    ax.tick_params(colors="#6b6760", labelsize=8.5, length=3)

    ax.fill_between(priors * 100, per_hit_model, per_hit_blind, color=GOLD, alpha=0.12)
    ax.plot(priors * 100, per_hit_blind, color=GREY, lw=2.0, label="不筛选：1 / 占比")
    ax.plot(priors * 100, per_hit_model, color=GOLD, lw=2.4, label="模型点名：1 / 命中率")

    for prior, dx, dy in ((0.01, 12, -16), (0.05, 8, 8)):
        value = 1.0 / ppv(prior, sn, sp)
        ax.scatter([prior * 100], [value], s=34, color=GOLD, zorder=3)
        ax.annotate(f"占 {prior:.0%}：{value:.1f} 条",
                    (prior * 100, value), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.6, color=INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks([1, 3, 10, 30, 100, 300], ["1", "3", "10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_xlabel("库中活性肽的占比（%）", fontsize=8.5, color="#6b6760")
    ax.set_ylabel("确认 1 条所需的合成数", fontsize=8.5, color="#6b6760")
    ax.grid(color=GREY_PALE, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sn", type=float, default=SN)
    parser.add_argument("--sp", type=float, default=SP)
    parser.add_argument("--peptide-cost", type=float, default=None,
                        help="synthesis plus assay for one peptide, in your currency")
    parser.add_argument("--hits", type=int, default=5, help="confirmed actives you want")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--wts-cost", type=float, default=None)
    parser.add_argument("--panel-cost", type=float, default=None)
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    report(args)
    if not args.no_chart:
        print(f"\nwrote {chart(sn=args.sn, sp=args.sp)}")


if __name__ == "__main__":
    main()
