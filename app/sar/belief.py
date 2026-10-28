"""The probability map (probability of area, POA, per cell) and its two updates: a look that found nobody, and time passing."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Belief:
    poa: np.ndarray  # (rows, cols), probability the subject is in each cell
    row: float  # rest of world: probability the subject is outside the area

    def total(self) -> float:
        return float(self.poa.sum() + self.row)

    def observe_miss(self, cells: tuple[slice, slice], pod: np.ndarray) -> float:
        """Bayes update after a photo of `cells` showed nobody. Returns the probability of success (POS) that look had.

        P(in cell | missed) = POA (1 - POD) / (1 - sum POA POD); ROW is renormalised the same way.
        """
        seen = self.poa[cells]
        pos = float((seen * pod).sum())
        if pos >= 1.0:
            raise ValueError("a look with POS 1 that found nobody contradicts the map")
        self.poa[cells] = seen * (1 - pod)
        scale = 1 / (1 - pos)
        self.poa *= scale
        self.row *= scale
        return pos

    def predict(self, move_fraction: float, passability: np.ndarray) -> None:
        """Time passes: `move_fraction` of each cell's mass moves to its 4 neighbours, weighted by their passability.

        Mass that walks off the edge of the area joins ROW. Total probability is conserved.
        """
        if move_fraction <= 0:
            return
        padded = np.pad(passability, 1, constant_values=1.0)  # the outside is reachable
        neighbour_sum = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        outflow = move_fraction * self.poa / neighbour_sum
        q = np.pad(outflow, 1)
        inflow = passability * (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:])
        moved = move_fraction * self.poa
        self.row += float(moved.sum() - inflow.sum())
        self.poa = self.poa - moved + inflow


def mobility(t_s: float, m0: float = 0.3, tau_s: float = 2 * 3600) -> float:
    """Fraction of probability mass that moves to a neighbouring cell per minute; decays as the subject tires or shelters."""
    return m0 * math.exp(-t_s / tau_s)
