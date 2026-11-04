# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""The README's headline chart: share of the 200 held-out searches found over the first 4 hours, from the committed
results (docs/results/final.json), for the expanding square, a volunteer who restarts it where the witness said, and
LastSeen. Run: uv run tools/plot_results.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results/final.json"
OUT = ROOT / "docs/img/found-over-time.png"
HOURS = 4
# planner key, label, colour, line style; the page's palette: ink for the baselines, red for what Meta's models add
LINES = [
    ("spiral", "Expanding square", "#56635a", "-"),
    ("spiral+witness", "Volunteer restarts it where the witness said", "#56635a", (0, (4, 3))),
    ("hedge+spark", "LastSeen (Muse Spark reads the reports)", "#da291c", "-"),
]
PAPER, INK, INK_3, RULE = "#f4f5f1", "#18241e", "#56635a", "#c9cfc5"


def found_curve(per_seed: list[dict], planner: str) -> tuple[list[float], list[float]]:
    """A step curve: hours, and the share of all searches found by then."""
    times = sorted(s["found_s"][planner] / 3600 for s in per_seed if s["found_s"][planner] is not None)
    xs, ys = [0.0], [0.0]
    for i, t in enumerate(times, 1):
        if t > HOURS:
            break
        xs += [t, t]
        ys += [ys[-1], i / len(per_seed)]
    return xs + [HOURS], ys + [ys[-1]]


def share_found(per_seed: list[dict], planner: str, hours: list[float]) -> list[float]:
    times = [s["found_s"][planner] / 3600 for s in per_seed if s["found_s"][planner] is not None]
    return [100 * sum(t <= h for t in times) / len(per_seed) for h in hours]


def half_found_h(per_seed: list[dict], planner: str) -> float:
    times = sorted(s["found_s"][planner] for s in per_seed if s["found_s"][planner] is not None)
    return times[len(per_seed) // 2 - 1] / 3600


def clock(hours: float) -> str:
    return f"{int(hours)} h {round(hours % 1 * 60):02d}"


def main() -> None:
    per_seed = json.loads(RESULTS.read_text(encoding="utf-8"))["per_seed"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for key, label, colour, style in LINES:
        xs, ys = found_curve(per_seed, key)
        ax.plot(xs, [y * 100 for y in ys], color=colour, linestyle=style, linewidth=2.6 if key == "hedge+spark" else 2,
                label=label, solid_capstyle="round")
    grid = [HOURS * i / 2000 for i in range(2001)]
    ours, square = share_found(per_seed, "hedge+spark", grid), share_found(per_seed, "spiral", grid)
    ax.fill_between(grid, square, ours, where=[a >= b for a, b in zip(ours, square)], step="post", color="#da291c",
                    alpha=0.10, linewidth=0, zorder=0)  # the time LastSeen gains on the square
    ax.axhline(50, color=RULE, linewidth=1, zorder=0)
    for key, colour, dy in (("hedge+spark", "#da291c", 7), ("spiral", INK_3, -11)):
        h = half_found_h(per_seed, key)
        ax.plot([h], [50], "o", color=colour, markersize=8, markeredgecolor=PAPER, markeredgewidth=2, zorder=5)
        ax.annotate(f"half found by {clock(h)}", (h, 50), xytext=(0, dy), textcoords="offset points", ha="center",
                    va="bottom" if dy > 0 else "top", color=colour, fontsize=11, fontweight="bold")
    ax.set_xlim(0, HOURS)
    ax.set_ylim(0, 70)
    ax.set_xticks(range(HOURS + 1), [f"{h} h" for h in range(HOURS + 1)])
    ax.set_yticks(range(0, 71, 10), [f"{y}%" for y in range(0, 71, 10)])
    ax.set_xlabel("Time since the search started", color=INK_3)
    ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=INK_3, length=0)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK, fontsize=10.5, handlelength=2.6)
    fig.suptitle("Missing people found, 200 held-out searches", x=0.035, ha="left", color=INK, fontsize=15,
                 fontweight="bold", y=0.97)
    fig.text(0.035, 0.875, "Simulated, one drone, scenarios fixed in advance and run once. Source: docs/results/final.json",
             color=INK_3, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.13)
    fig.savefig(OUT, facecolor=PAPER, dpi=200)
    print(OUT.relative_to(ROOT), {key: clock(half_found_h(per_seed, key)) for key, *_ in LINES})


if __name__ == "__main__":
    main()
