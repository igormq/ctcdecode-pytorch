from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Generic, MutableMapping, Tuple, TypeVar

import numpy as np
from ctcdecode.csrc import binding

print(5)

PathTrie = binding.PathTrie


def topk(root, k):
    prefixes = binding.ListTrie()

    # print(prefixes[0])
    root.iterate_to_vec(prefixes)

    # print(1, k, len(prefixes))
    binding.prune(prefixes, k)
    # print(2, k, len(prefixes))
    return prefixes


def sort(prefixes):
    return sorted(prefixes, key=lambda x: -x.score)


# T = TypeVar('T')
# @dataclass
# class PathTrie(Generic[T]):
#     p_b: float = -float('inf')
#     p_nb: float = -float('inf')

#     n_p_b: float = -float('inf')
#     n_p_nb: float = -float('inf')

#     p: float = -float('inf')

#     score: float = -float('inf')
#     score_lm: float = 0.0
#     score_ctc: float = -float('inf')

#     exists: bool = True

#     parent: T = None
#     children: MutableMapping[str, T] = field(default_factory=dict)

#     prefix: Tuple[int, ...] = field(default_factory=tuple)

#     state: MutableMapping[str, Any] = field(default_factory=dict)
#     timesteps: Tuple[int, ...] = field(default_factory=tuple)

#     def step(self):
#         self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
#         self.n_p_b = self.n_p_nb = -float('inf')
#         self.score_ctc = np.logaddexp(self.p_b, self.p_nb)
#         self.score = self.score_ctc + self.score_lm

#     def reset(self, timestep=None, p=-float('inf')):
#         if not self.exists:
#             self.exists = True
#             self.p_b = self.p_nb = self.n_p_b = self.n_p_nb = -float('inf')

#         if timestep and self.p < p:
#             self.timestep = timestep
#             self.p = p

#     def get_path_trie(self, key, timestep=None, p=-float('inf')):
#         if key in self.children:
#             node = self.children[key]
#             node.reset(timestep=timestep, p=p)
#             return node

#         node = PathTrie(prefix=self.prefix + (key, ),
#                         parent=self,
#                         timesteps=self.timesteps + (timestep, ))
#         self.children[key] = node

#         return node

#     def __iter__(self):
#         return self.iter()

#     def iter(self, step=True):
#         if self.exists:
#             if step:
#                 self.step()
#             yield self

#         for child in self.children.values():
#             yield from iter(child)

#     def remove(self):
#         self.exists = False

#         if not len(self.children):
#             del self.parent.children[self.prefix[-1]]

#         if self.parent and not len(self.parent.children) and not self.parent.exists:
#             self.parent.remove()

#         del self

#     def __repr__(self):
#         return (
#             f'Node(key={self.prefix[-1] if self.prefix else -1}, timestep={self.timesteps[-1] if self.timesteps else -1}, children={len(self.children)}, p_b={self.p_b:.4f}, p_nb={self.p_nb:.4f}, '
#             f'score={self.score:.4f})')

#     def __del__(self):
#         for child in self.children:
#             del child

# def topk(root, k, step=True):
#     """ Keep only the top k prefixes """

#     nodes = list(root.iter(step))

#     if len(nodes) <= k:
#         return nodes

#     indexes = np.argpartition([-n.score for n in nodes], k).tolist()

#     for n in itemgetter(*indexes[k:])(nodes):
#         n.remove()

#     return itemgetter(*indexes[:k])(nodes)

# def sort(nodes):
#     indexes = np.argsort([-n.score for n in nodes]).tolist()
#     return itemgetter(*indexes)(nodes)
