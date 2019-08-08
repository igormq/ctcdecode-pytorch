import time

import torch

from ctcdecode.csrc import _C
from ctcdecode.decoders import ctc_beam_search_decoder
from ctcdecode.lm import KenLM
from ctcdecode.tokenizer import Tokenizer

c = Tokenizer('tests/data/toy-data-vocab.txt')
print(c.get_index("<space>"))
print(c.get_index("<blank>"))

c.space_index = c.get_index("<space>")
c.blank_index = c.get_index("<blank>")

lm = KenLM('tests/data/bigram.arpa', c)
print(1)

log_probs = torch.randn(100, 29).log_softmax(1)

# s = lm.start(False)
# print(s)

# print('space_idx', c.get_space_idx())

# print(c.entry2idx('t'))
# s, score = lm.score(s, c.entry2idx('t')[0])
# s, score = lm.score(s, c.entry2idx('h')[0])
# s, score = lm.score(s, c.entry2idx('e')[0])
# s, score = lm.score(s, c.entry2idx(['<space>'])[0])
# s, score = lm.finish(s)
# print(score)
s = time.time()
o = ctc_beam_search_decoder(log_probs, lm_scorer=lm)
print(time.time() - s)
# s = _C.start(lm)

# # print(s.tokens)
# print(s)
# print(_C.to_py_object(s))

# print(2)
