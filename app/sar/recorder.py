"""The event log of one run, for replay in the UI. Everything the page shows comes from here.

Events (all with `t_s`, search time in seconds):
- `drones`: positions [y_m, x_m, returning] of every drone, every FRAME_S;
- `photo`: the cells one photo covered, [row0, row1, col0, col1) (row1 and col1 excluded);
- `report`: witness text as the team received it;
- `note`: a planner explaining itself (e.g. Spark's hypotheses, why a block was chosen), if the planner offers notes;
- `map`: the planner's probability map, shrunk to MAP_SIDE x MAP_SIDE and scaled to 0-255, every MAP_EVERY_S;
- `forced_return`, `found`.
"""

import numpy as np

FRAME_S = 30.0
MAP_EVERY_S = 300.0
MAP_SIDE = 60


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self._next_frame = 0.0
        self._next_map = 0.0

    def tick(self, t_s: float, drones, planner) -> None:
        """Called every arena step; records only when a frame or map snapshot is due, plus any planner notes."""
        drain = getattr(planner, "drain_notes", None)
        for note in drain() if drain else ():
            self.events.append({"type": "note", "t_s": t_s, **note})
        if t_s >= self._next_frame:
            self.events.append({"type": "drones", "t_s": t_s,
                                "drones": [[round(d.y, 1), round(d.x, 1), d.returning] for d in drones]})
            self._next_frame += FRAME_S
        if t_s >= self._next_map:
            poa = planner.probability_map()
            if poa is not None:
                self.events.append({"type": "map", "t_s": t_s, "side": MAP_SIDE, "values": _shrink(poa)})
            self._next_map += MAP_EVERY_S

    def add(self, kind: str, t_s: float, **fields) -> None:
        self.events.append({"type": kind, "t_s": t_s, **fields})


def _shrink(poa: np.ndarray) -> list[int]:
    """Sum the map into MAP_SIDE x MAP_SIDE blocks and scale to 0-255 (relative to the fullest block)."""
    rows = np.array_split(np.arange(poa.shape[0]), MAP_SIDE)
    cols = np.array_split(np.arange(poa.shape[1]), MAP_SIDE)
    blocks = np.array([[poa[np.ix_(r, c)].sum() for c in cols] for r in rows])
    top = blocks.max()
    scaled = np.zeros_like(blocks) if top <= 0 else blocks / top * 255
    return [int(v) for v in scaled.round().ravel()]
