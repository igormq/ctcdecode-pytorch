import torch
from ctcdecode.csrc import _C


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
def ctc_beam_search_decoder(log_probs_seq, seq_lengths=None, lm=None, beam_size=100, blank=0, cutoff_prob=1.0, cutoff_top_n=40, alpha=0.0, beta=0.0, num_processes=1):
    """
    Performs prefix beam search on the output of a CTC network.

    Args:
        log_probs_seq (tensor): The log probabilities. Should be a 3D array (batch_size x timesteps x alphabet_size)
        lm (func): Language model function. Should take as input a string and output a
            probability.
        beam_size (int): The beam width. Will keep the `beam_size` most likely candidates at each
            timestep.
        blank (int): Blank label index
        cutoff_prob: Cutoff probability for pruning. Defaults to `1.0`, meaning no pruning
        cutoff_top_n: Cutoff number for pruning.

    Retruns:
        string: The decoded CTC output.
    """
    log_probs_seq = log_probs_seq.float()

    if not seq_lengths:
        batch_size = log_probs_seq.shape[0]
        max_time = log_probs_seq.shape[1]
        seq_lengths = torch.tensor([max_time for _ in range(batch_size)], dtype=torch.int)
    
    seq_lengths = seq_lengths.int()

    beam_result = _C.beam_decoder_batch(log_probs_seq, seq_lengths, blank, beam_size, num_processes, cutoff_prob,
                                  cutoff_top_n, lm, alpha, beta)

    return [(beam_result[1][0][b].item(), beam_result[0][0][b][:beam_result[3][0][b].item()].tolist(),
             beam_result[2][0][b][:beam_result[3][0][b].item()].tolist()) for b in range(beam_size)]
