top
import time

import torch
import os
import numpy as np

from ctcdecode.csrc import _C
from ctcdecode.decoders import ctc_beam_search_decoder
from ctcdecode.lm import KenLM, KenLMUnit
from ctcdecode.tokenizer import Tokenizer

data_dir = 'tests/data'

data = np.genfromtxt(os.path.join(data_dir, "rnn_output.csv"), delimiter=';')[:, :-1]
inputs = torch.as_tensor(data).log_softmax(1)

tokenizer = Tokenizer(os.path.join(data_dir, 'labels.txt'))
tokenizer.blank_index = tokenizer.get_index('<blank>')
tokenizer.space_index = tokenizer.get_index('<space>')

# lm-based decoding
scorer = KenLM(os.path.join(data_dir, 'bigram.arpa'), tokenizer, unit=KenLMUnit.Word)
s = time.time()
result = ctc_beam_search_decoder(inputs, lm_scorer=scorer, blank=tokenizer.blank_index, beam_size=25, alpha=2, beta=0)
e = time.time()
txt_result = ''.join(tokenizer.idxs2entries(result[0][1])).replace('<space>', ' ')
print("Expected: the fake friend of the family, lie th")
print("Got     : " + txt_result)
print("Decoder time: {} s".format(e-s))
