import os

import numpy as np
import copy
import kenlm
from ctcdecode.fst import FST


class KenLMScorer:
    """External scorer to evaluate a prefix or whole sentence in
       beam search decoding, including the score from n-gram language
       model and word count using the KenLM

    Args:
        model_path: Path to load language model.
        vocab: list of characters that maps the network outputs to its corresponding character
    """
    LOG10_E = np.log10(np.exp(1))
    OOV_SCORE = -1000.0

    def __init__(self, model_path, tokenizer, fst_path=None, alpha=0, beta=0, unit='word'):

        self.lm = kenlm.Model(model_path)
        self.fst = None
        self.tokenizer = tokenizer

        self.order = self.lm.order

        self.alpha = alpha
        self.beta = beta

        self.unit = unit

        self.fst = None
        if self.unit == 'word':
            if fst_path:
                self.fst = FST(fst_path)

    def __call__(self, token_ids, state, eos=False):
        """ Evaluation function
        """
        sentence = self.tokenizer.idx2token(token_ids)

        if self.unit == 'word':
            tokens = sentence.split(' ')
            if not eos:
                if tokens[-1] != '':  # if not EOS, it must end with a space
                    return 0.0

                tokens = tokens[:-1]
        else:
            tokens = list(map(lambda x: x.replace(' ', '<space>'), list(sentence)))

        # TODO: deepcopy of kenlm.State
        # bos = True if 'lm_state' not in state else False
        # state.setdefault('score', 0.0)
        # lm_state = state.get('lm_state', kenlm.State())
        # lm_state_out = kenlm.State()

        # if bos:
        #     self.lm.BeginSentenceWrite(lm_state)

        # score = self.lm.BaseScore(lm_state, tokens[-1], lm_state_out)
        # lm_state, lm_state_out = lm_state_out, lm_state

        # if eos:
        #     score += self.lm.BaseScore(lm_state, '</s>', lm_state_out)
        #     lm_state, lm_state_out = lm_state_out, lm_state

        # state['lm_state'] = lm_state

        sentence = ' '.join(tokens)

        if not len(sentence) and not eos:
            return 0.0

        scores = list(self.lm.full_scores(sentence, eos=eos))

        is_oov = scores[-1][-1] | scores[-2][-1] if eos and self.unit == 'word' else scores[-1][-1]
        score = scores[-1][0] + scores[-2][0] if eos and self.unit == 'word' else scores[-1][0]

        if is_oov:
            return self.OOV_SCORE

        # print(bos, eos, tokens[-1], '`' + ' '.join(tokens) + '`', total_score, state['score'], curr_score)
        # state['history'].append([bos, eos, tokens[-1], '`' + ' '.join(tokens) + '`', total_score, state['score'], curr_score])

        # if not np.allclose(total_score, state['score']):
        #     print('cu de historico')
        #     for s in state['history']:
        #         print(s)
        #     print('end')
        # assert np.allclose(total_score, state['score'])

        # print('toma')
        # print(self.alpha * total_score / self.LOG10_E + self.beta * len(tokens))
        return self.alpha * score / self.LOG10_E + self.beta

    def is_valid(self, token, state):

        if not self.fst:
            return True

        state.setdefault('fst_state', self.fst.initial_state)

        self.fst.set_state(state['fst_state'])

        found = self.fst.find(token)

        if not found:
            return False

        state['fst_state'] = self.fst.next_state

        if self.fst.is_final(state['fst_state']):
            state['fst_state'] = self.fst.initial_state

        return True
