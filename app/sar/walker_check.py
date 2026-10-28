"""Check the truth walker against published lost-person statistics: walk many subjects on synthetic maps and compare
where they end up with the distance rings, dispersion angles and track offsets.

    uv run python -m app.sar.walker_check --category dementia --maps 20 --per-map 50
"""

import argparse
import dataclasses

import numpy as np

from app.sar.scenario import synthetic
from app.sar.stats import lost_person_stats
from app.sar.walker import PARAMS, walk

END_H = 30.0  # long enough for nearly every subject to stop moving
PCT = (25, 50, 75, 95)


def end_points(category: str, maps: int, per_map: int, params) -> dict[str, np.ndarray]:
    dist, angle, offset = [], [], []
    for seed in range(maps):
        m = synthetic(10_000 + seed, size=320, category=category).map  # 8 km square, so most walkers stay mapped
        w = walk(m, category, np.random.default_rng(seed), per_map, END_H * 3600, params)
        start = np.array(m.grid.centre_m(*m.ipp))
        delta = w.positions[-1] - start
        d = np.hypot(*delta.T)
        dist.append(d / 1000)
        bearing = np.arctan2(delta[:, 0], delta[:, 1])
        moved = d > 50
        angle.append(np.degrees(np.abs(np.angle(np.exp(1j * (bearing - w.start_heading)))))[moved])
        offset.append(_distance_to_trail(m, w.positions[-1]))
    return {"dist_km": np.concatenate(dist), "dispersion_deg": np.concatenate(angle), "track_offset_m": np.concatenate(offset)}


def _distance_to_trail(m, points: np.ndarray) -> np.ndarray:
    ty, tx = np.nonzero(m.trails)
    trail_m = (np.stack([ty, tx], axis=1) + 0.5) * m.grid.cell_m
    return np.min(np.hypot(points[:, None, 0] - trail_m[None, :, 0], points[:, None, 1] - trail_m[None, :, 1]), axis=1)


def report(category: str, result: dict[str, np.ndarray]) -> float:
    """Print simulated vs published percentiles; return the mean absolute log error on distance."""
    published = lost_person_stats()["categories"][category]
    rows = [("distance km", result["dist_km"], published["rings_km"]),
            ("dispersion deg", result["dispersion_deg"], published.get("dispersion_deg")),
            ("track offset m", result["track_offset_m"], published.get("track_offset_m"))]
    print(f"{category}: {len(result['dist_km'])} walkers   percentiles {PCT}")
    for label, values, target in rows:
        sim = np.percentile(values, PCT)
        print(f"  {label:<16} sim {_fmt(sim)}   published {_fmt(target) if target else '-'}")
    err = float(np.mean(np.abs(np.log(np.percentile(result["dist_km"], PCT) / np.array(published["rings_km"])))))
    print(f"  distance error (mean |log ratio|): {err:.2f}")
    return err


def _fmt(values) -> str:
    return " ".join(f"{v:>6.2f}" if v is not None else "     -" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", default="dementia", choices=sorted(PARAMS))
    parser.add_argument("--maps", type=int, default=20)
    parser.add_argument("--per-map", type=int, default=50)
    parser.add_argument("--speed", type=float, help="override walking speed (m/s) to explore the fit")
    parser.add_argument("--moving-h", type=float, help="override median moving time (h)")
    parser.add_argument("--turn", type=float, help="override heading drift (deg per minute)")
    args = parser.parse_args()
    params = PARAMS[args.category]
    overrides = {k: v for k, v in (("speed_mps", args.speed), ("moving_h", args.moving_h), ("turn_sd_deg", args.turn)) if v is not None}
    params = dataclasses.replace(params, **overrides)
    report(args.category, end_points(args.category, args.maps, args.per_map, params))


if __name__ == "__main__":
    main()
