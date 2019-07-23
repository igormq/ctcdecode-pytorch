import torch
import numpy as np
from collections import OrderedDict as odict
import copy


class Beam:
    def __init__(self, state={}, timesteps=(0,)):
        self.p_b = self.p_nb = -float('inf')
        self.n_p_b = self.n_p_nb = -float('inf')

        self.score_lm = 0.0
        self.score_ctc = -float('inf')

        self.state = state
        self.timesteps = timesteps

    def step(self):
        self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
        self.n_p_b = self.n_p_nb = -float('inf')
        self.score_ctc = np.logaddexp(self.p_b, self.p_nb)

    @property
    def score(self):
        return self.score_ctc + self.score_lm

    def copy(self, timestep=(0,)):
        if isinstance(timestep, int):
            timestep = (timestep, )

        beam = Beam(timesteps=self.timesteps + timestep)
        beam.state = copy.deepcopy(self.state)
        return beam

    def __repr__(self):
        return f'Beam(p_b={self.p_b:.4f}, p_nb={self.p_nb:.4f}, score={self.score:.4f})'


class Beams:
    def __init__(self):
        self.beams = odict()
        self.beams[()] = Beam()
        self.beams[()].p_b = 0
        self.beams[()].score_ctc = 0.0

    def __getitem__(self, prefix):
        return self.get(prefix)

    def get(self, prefix, timestep=None, is_valid=None):

        if prefix in self.beams:
            return self.beams[prefix]

        timesteps = (0, )
        if len(prefix):
            new_beam = self.beams[prefix[:-1]].copy(timestep=timestep)
            if is_valid and not is_valid(prefix[-1], new_beam.state):
                return None
        else:
            new_beam = Beam(timesteps=timesteps)
        self.beams[prefix] = new_beam

        return new_beam

    def items(self):
        return self.beams.items()

    def __len__(self):
        return len(self.beams)

    def __iter__(self):
        return iter(self.beams)

    def keys(self):
        return self.beams.keys()

    def values(self):
        return self.beams.values()

    def step(self):
        for beam in self.beams.values():
            beam.step()

    def topk_(self, k):
        """ Keep only the top k prefixes """
        if len(self.beams) <= k:
            return self

        beam_scores = torch.as_tensor([beam.score for beam in self.beams.values()])
        _, indexes = torch.topk(beam_scores, k)

        self.beams = odict((k, v) for i, (k, v) in enumerate(self.beams.items()) if i in indexes)

        return self

    def sort(self):
        return sorted(self.beams.items(), key=lambda x: -x[1].score)


    def __delitem__(self, key):
        self.beams.__delitem__(key)

    def __setitem__(self, key, value):
        self.beams[key] = value
