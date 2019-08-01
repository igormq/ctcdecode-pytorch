import math

import numpy as np
import torch

from .beams import Beams


def logcumsumexp(x, dim=0):
    m, _ = x.max(dim=dim, keepdim=True)
    y = (x - m).exp()
    return torch.log(torch.cumsum(y, dim=dim)) + m


@torch.no_grad()
def ctc_greedy_decoder(log_probs_seq, blank=0):
    """CTC greedy (best path) decoder.

    Path consisting of the most probable tokens are further post-processed to
    remove consecutive repetitions and all blanks.

    Args:
        log_probs_seq: 2-D tensor containing the log probability of a character given each timestep
        blank: blank label index. Defaults to 0
    Returns:
        tuple containing (score, decoded sequence, timesteps)
    """
    # argmax to get the best index for each time step
    max_probs, max_indexes = torch.max(log_probs_seq, 1)
    # remove consecutive duplicate indexes
    mask = torch.cat([
        torch.tensor([1], dtype=torch.uint8, device=log_probs_seq.device),
        ((max_indexes[:-1] - max_indexes[1:]).abs() > 0)
    ])
    # remove blank indexes
    mask = mask * (max_indexes != blank)

    return -max_probs.sum(), max_indexes[mask], mask.nonzero().squeeze()


@torch.no_grad()
def ctc_beam_search_decoder(log_probs_seq,
                            lm_scorer=None,
                            beam_size=100,
                            blank=0,
                            cutoff_prob=1.0,
                            cutoff_top_n=None):
    """
    Performs prefix beam search on the output of a CTC network.

    Args:
        log_probs_seq (tensor): The log probabilities. Should be a 2D array (timesteps x
            alphabet_size)
        lm_scorer (func): Language model function. Should take as input a string and output a
            probability.
        beam_size (int): The beam width. Will keep the `beam_size` most likely candidates at each
            timestep.
        blank (int): Blank label index
        cutoff_prob: Cutoff probability for pruning. Defaults to `1.0`, meaning no pruning
        cutoff_top_n: Cutoff number for pruning.

    Retruns:
        string: The decoded CTC output.
    """
    T, V = log_probs_seq.shape
    log_cutoff_prob = math.log(cutoff_prob)
    cutoff_top_n = min(cutoff_top_n, V) if cutoff_top_n else V

    beams = Beams(is_valid=lm_scorer.is_valid if lm_scorer else None)

    for t in range(T):

        log_probs = log_probs_seq[t]

        curr_beams = list(beams.sort())

        # A default dictionary to store the next step candidates.
        num_prefixes = len(curr_beams)

        min_cutoff = -float('inf')
        full_beam = False

        if lm_scorer:
            # score - beta # it will not insert a new word or character
            min_cutoff = curr_beams[-1][-1].score_ctc + log_probs[blank]
            full_beam = num_prefixes == beam_size

        # Prunning step
        pruned_indexes = torch.arange(len(log_probs)).tolist()
        if log_cutoff_prob < 0.0 or cutoff_top_n < V:
            idxs = torch.argsort(log_probs, descending=True)
            n_idxs = min((logcumsumexp(log_probs[idxs], 0) <= log_cutoff_prob).sum(), cutoff_top_n,
                         V)
            pruned_indexes = idxs[:n_idxs].tolist()

        for token_index in pruned_indexes:

            p = log_probs[token_index].item()

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, beam in curr_beams:
                p_b, p_nb = beam.p_b, beam.p_nb

                if full_beam and p + beam.score_ctc < min_cutoff:
                    break

                # If we propose a blank the prefix doesn't change. Only the probability of ending
                # in blank gets updated.
                if token_index == blank:
                    beam.n_p_b = np.logaddexp(beam.n_p_b, beam.score_ctc + p)
                    continue

                # Extend the prefix by the new character s and add it to the beam[' Only'] the
                # probability of not ending in blank gets updated.
                last_token_index = prefix[-1] if prefix else None

                if token_index == last_token_index:
                    # If s is repeated at the end we also update the unchanged prefix. This is the
                    # merging case.
                    beam.n_p_nb = np.logaddexp(beam.n_p_nb, p_nb + p)

                n_prefix = prefix + (token_index, )

                # Must update state for prefix search
                n_beam = beams.getitem(n_prefix, p=p, previous_beam=beam)
                if not n_beam:
                    continue

                n_p_nb = n_beam.n_p_nb

                if token_index == last_token_index and p_b > -float('inf'):
                    # We don't include the previous probability of not ending in blank (p_nb)
                    # if s is repeated at the end. The CTC algorithm merges characters not
                    # separated by a blank.
                    n_p_nb = np.logaddexp(n_p_nb, p_b + p)
                elif token_index != last_token_index:
                    n_p_nb = np.logaddexp(n_p_nb, beam.score_ctc + p)

                if lm_scorer:
                    # LM scorer has access and updates the state variabl
                    p_lm = lm_scorer(n_prefix, n_beam.state)
                    n_beam.score_lm = beam.score_lm + p_lm

                n_beam.n_p_nb = n_p_nb

        # Update the probabilities
        beams.step()
        # Trim the beam before moving on to the next time-step.
        beams.topk_(beam_size)

    # score the eos
    if lm_scorer:
        for prefix, beam in beams.items():
            if prefix:
                p_lm = lm_scorer(prefix, beam.state, eos=True)
                beam.score_lm += p_lm
                beam.score = beam.score_ctc + beam.score_lm

    # Return the top beam_size -log probabilities without the lm scoring
    # return [(-beam['score_ctc'], p, beam['timesteps']) for p, beam in beams.sort()]
    return [(-beam.score, p, beam.timesteps) for p, beam in beams.sort()]
