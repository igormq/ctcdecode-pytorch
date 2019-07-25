from collections.abc import MutableMapping
from dataclasses import dataclass
from operator import itemgetter
from typing import Tuple

import numpy as np

LOG_0 = -float('inf')
LOG_1 = 0.0


@dataclass
class Beam:
    p_b: float = LOG_0
    p_nb: float = LOG_0

    n_p_b: float = LOG_0
    n_p_nb: float = LOG_0

    score: float = LOG_0
    score_lm: float = LOG_1
    score_ctc: float = LOG_0

    state = {}
    timesteps: Tuple = ()

    def step(self):
        self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
        self.n_p_b = self.n_p_nb = LOG_0
        self.score_ctc = np.logaddexp(self.p_b, self.p_nb)
        self.score = self.score_ctc + self.score_lm

    def __repr__(self):
        return (f'Beam(p_b={self.p_b:.4f}, p_nb={self.p_nb:.4f}, ' f'score={self.score:.4f})')


class Beams(MutableMapping):
    def __init__(self, is_valid=None):
        self.is_valid = is_valid
        self.timestep = 0

        self.beams = {(): Beam()}
        self.beams[()].p_b = 0
        self.beams[()].score_ctc = 0.0

    def __getitem__(self, key):
        return self.getitem(key)

    def getitem(self, key, previous_beam=None):

        if key in self.beams:
            return self.beams[key]

        new_beam = Beam()

        if previous_beam:
            new_beam.timesteps = previous_beam.timesteps + (self.timestep, )
            new_beam.state = previous_beam.state

            if self.is_valid and not self.is_valid(key[-1], new_beam.state):
                return None

        self.beams[key] = new_beam

        return new_beam

    def __setitem__(self, key, value):
        self.beams[key] = value

    def __delitem__(self, key):
        del self.beams[key]

    def __len__(self):
        return len(self.beams)

    def __iter__(self):
        return iter(self.beams)

    def step(self):

        for beam in self.beams.values():
            beam.step()

        self.timestep += 1

    def topk_(self, k):
        """ Keep only the top k prefixes """
        if len(self.beams) <= k:
            return self

        beams = list(self.beams.items())
        indexes = np.argpartition([-v.score for k, v in beams], k)[:k].tolist()

        self.beams = {k: v for k, v in itemgetter(*indexes)(beams)}

        return self

    def sort(self):
        return sorted(self.beams.items(), key=lambda x: x[1].score, reverse=True)
