import numpy as np

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

    def __init__(self, model_path, tokenizer, trie_path=None, alpha=0, beta=0, unit='word'):

        self.lm = kenlm.Model(model_path)
        self.fst = None
        self.tokenizer = tokenizer

        self.order = self.lm.order

        self.alpha = alpha
        self.beta = beta

        self.unit = unit

        self.fst = None
        if self.unit == 'word':
            if trie_path:
                self.fst = FST(trie_path)

    def __call__(self, token_ids, state, eos=False):
        """ Evaluation function
        """

        sentence = self.tokenizer.idx2token(token_ids)

        if self.unit == 'word':
            # if not EOS, it must end with a space
            if not eos and token_ids[-1] != self.tokenizer.space_idx:
                return 0.0

            tokens = sentence.split()
        else:
            tokens = list(map(lambda x: x.replace(' ', '<space>'), list(sentence)))

        sentence = ' '.join(tokens).strip()

        if not sentence:
            return self.OOV_SCORE

        scores = list(self.lm.full_scores(sentence, eos=eos))

        is_oov = scores[-1][-1] | scores[-2][-1] if eos and self.unit == 'word' and token_ids[
            -1] != self.tokenizer.space_idx else scores[-1][-1]
        score = scores[-1][0] + scores[-2][0] if eos and self.unit == 'word' and token_ids[
            -1] != self.tokenizer.space_idx else scores[-1][0]

        if is_oov and not eos:
            return self.OOV_SCORE

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
