import copy

import numpy as np

import kenlm
from ctcdecode.csrc import _C

print(2)

KenLM = _C.KenLM
KenLMUnit = _C.KenLMUnit
# class KenLMState(_C.LMState):
#     __slots__ = ['state', 'tokens']

#     def __init__(self, state, tokens):
#         self.state = state
#         self.tokens = tokens

#     def __del__(self):
#         print('I am gone')


# class KenLM(_C.LM):
#     """External scorer to evaluate a prefix or whole sentence in
#        beam search decoding, including the score from n-gram language
#        model and word count using the KenLM

#     Args:
#         model_path: Path to load language model.
#         vocab: list of characters that maps the network outputs to its corresponding character
#     """
#     LOG10_E = np.log10(np.exp(1))
#     OOV_SCORE = -1000.0

#     def __init__(self, model_path, tokenizer, alpha=0, beta=0, unit='word'):
#         super().__init__()
#         # _C.LM.__init__(self)

#         self.lm = kenlm.Model(model_path)
#         self.order = self.lm.order

#         self.alpha = alpha
#         self.beta = beta

#         self.unit = unit

#         self.states = []

#     def start(self, start_with_nothing):
#         state = kenlm.State()

#         if not start_with_nothing:
#             self.lm.BeginSentenceWrite(state)
#         else:
#             self.lm.NullSentenceWrite(state)

#         s = KenLMState(state=state, tokens=[])
#         self.states.append(s)

#         return _C.to_shared_ptr(len(self.states) - 1)

#     def score(self, state_index, token_index):
#         state = self.state[state_index]

#         if token_index != self.tokenizer.space_index:
#             state.state += [token_index]
#             return state_index, -float('inf')

#         word = self.tokenizer.idx2token(state.tokens)

#         new_state = kenlm.State()
#         score = self.lm.BaseScore(state.state, word, new_state)

#         if word not in self.lm:
#             score = self.OOV_SCORE

#         self.state.append(state)
#         return len(self.state) - 1, score

#     def finish(self, state_index):
#         state = self.state[state_index]

#         score = 0.0
#         if len(state.tokens):
#             state_index, score = self.score(state_index, self.tokenizer.space_index)
#             state = self.state[state_index]

#         new_state = kenlm.State()
#         score += self.lm.BaseScore(state.state, '</s>', new_state)

#         self.state.append(KenLMState(state=new_state, tokens=[]))
#         return len(self.state) - 1, score

#     def compare_state(self, state1, state2):
#         if state1.state == state2.state:
#             return 0

#         return state1.state > state2.state

#     # def __call__(self, token_ids, state, eos=False):
#     #     """ Evaluation function
#     #     """
#     #     sentence = self.tokenizer.idx2token(token_ids)

#     #     if self.unit == 'word':
#     #         tokens = sentence.split(' ')
#     #         if not eos:
#     #             if tokens[-1] != '':  # if not EOS, it must end with a space
#     #                 return 0.0

#     #             tokens = tokens[:-1]
#     #     else:
#     #         tokens = list(map(lambda x: x.replace(' ', '<space>'), list(sentence)))

#     #     # TODO: deepcopy of kenlm.State
#     #     # bos = True if 'lm_state' not in state else False
#     #     # state.setdefault('score', 0.0)
#     #     # lm_state = state.get('lm_state', kenlm.State())
#     #     # lm_state_out = kenlm.State()

#     #     # if bos:
#     #     #     self.lm.BeginSentenceWrite(lm_state)

#     #     # score = self.lm.BaseScore(lm_state, tokens[-1], lm_state_out)
#     #     # lm_state, lm_state_out = lm_state_out, lm_state

#     #     # if eos:
#     #     #     score += self.lm.BaseScore(lm_state, '</s>', lm_state_out)
#     #     #     lm_state, lm_state_out = lm_state_out, lm_state

#     #     # state['lm_state'] = lm_state

#     #     sentence = ' '.join(tokens)

#     #     if not len(sentence) and not eos:
#     #         return 0.0

#     #     scores = list(self.lm.full_scores(sentence, eos=eos))

#     #     is_oov = scores[-1][-1] | scores[-2][-1] if eos and self.unit == 'word' else scores[-1][-1]
#     #     score = scores[-1][0] + scores[-2][0] if eos and self.unit == 'word' else scores[-1][0]

#     #     if is_oov:
#     #         return self.OOV_SCORE

#     #     # print(bos, eos, tokens[-1], '`' + ' '.join(tokens) + '`', total_score, state['score'], curr_score)
#     #     # state['history'].append([bos, eos, tokens[-1], '`' + ' '.join(tokens) + '`', total_score, state['score'], curr_score])

#     #     # if not np.allclose(total_score, state['score']):
#     #     #     print('cu de historico')
#     #     #     for s in state['history']:
#     #     #         print(s)
#     #     #     print('end')
#     #     # assert np.allclose(total_score, state['score'])

#     #     # print('toma')
#     #     # print(self.alpha * total_score / self.LOG10_E + self.beta * len(tokens))
#     #     return self.alpha * score / self.LOG10_E + self.beta

#     # def is_valid(self, token, state):

#     #     if not self.trie:
#     #         return True

#     #     state.setdefault('trie_state', self.trie.initial_state)

#     #     self.trie.set_state(state['trie_state'])

#     #     found = self.trie.find(token)

#     #     if not found:
#     #         return False

#     #     state['trie_state'] = self.trie.next_state

#     #     if self.trie.is_final(state['trie_state']):
#     #         state['trie_state'] = self.trie.initial_state

#     #     return True
