import argparse
import logging
from functools import partial
from itertools import product
from timeit import default_timer

import pandas as pd
import torch
import tqdm
from torch.multiprocessing import Pool, cpu_count

from ctcdecode import metrics
from ctcdecode.decoders import ctc_beam_search_decoder
from ctcdecode.lm import KenLM, LMUnit
from ctcdecode.tokenizer import Tokenizer

LOG = logging.getLogger('tune')


def tune_from_args(args):

    params_grid = list(
        product(torch.linspace(args.alpha_from, args.alpha_to, args.alpha_steps),
                torch.linspace(args.beta_from, args.beta_to, args.beta_steps)))

    LOG.info('Scheduling {} jobs for alphas=linspace({}, {}, {}), betas=linspace({}, {}, {})'.format(
        len(params_grid), args.alpha_from, args.alpha_to, args.alpha_steps, args.beta_from, args.beta_to,
        args.beta_steps))

    # start worker processes
    LOG.info(f"Using {args.num_workers} processes and {args.lm_workers} for each CTCDecoder.")
    extract_start = default_timer()

    tokenizer = Tokenizer(args.vocab_file)
    tokenizer.space_index = tokenizer.get_index('<space>')
    tokenizer.blank_index = tokenizer.get_index('<blank>')

    p = Pool(args.num_workers, init, [
        args.logits_targets_file, args.vocab_file, args.lm_path, args.lm_trie_path, args.lm_unit, args.beam_size,
        args.cutoff_prob, args.cutoff_top_n
    ])

    scores = []
    best_wer = float('inf')
    with tqdm.tqdm(p.imap(tune_step, params_grid), total=len(params_grid), desc='Grid search') as pbar:
        for i, params in enumerate(pbar):
            alpha, beta, wer, cer = params
            scores.append([alpha, beta, wer, cer])

            if wer < best_wer:
                best_wer = wer
                pbar.set_postfix(alpha=alpha, beta=beta, wer=wer, cer=cer)

            if i % 10 == 0:
                df = pd.DataFrame(scores, columns=['alpha', 'beta', 'wer', 'cer'])
                df.to_csv(args.output_file, index=False)

    df = pd.DataFrame(scores, columns=['alpha', 'beta', 'wer', 'cer'])
    df.to_csv(args.output_file, index=False)

    LOG.info(f"Finished {len(params_grid)} processes in {default_timer() - extract_start:.1f}s")


def init(logits_targets_file, vocab_file, lm_path, trie_path, lm_unit, beam_size, cutoff_prob, cutoff_top_n):
    global saved_outputs
    global decoder
    global tokenizer
    global lm

    saved_outputs = torch.load(logits_targets_file)
    tokenizer = Tokenizer(vocab_file)
    lm = KenLM(lm_path, tokenizer, unit=LMUnit.Word if lm_unit == 'word' else LMUnit.Char, trie_path=trie_path)
    decoder = partial(ctc_beam_search_decoder,
                      beam_size=beam_size,
                      blank=tokenizer.blank_index,
                      cutoff_prob=cutoff_prob,
                      cutoff_top_n=cutoff_top_n)


def tune_step(params):
    alpha, beta = params
    alpha = alpha.item()
    beta = beta.item()

    global tokenizer
    global decoder
    global saved_outputs
    global lm

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    for i, (output, reference) in enumerate(saved_outputs):

        transcript = decoder(torch.as_tensor(output, dtype=torch.float32), alpha=alpha, beta=beta)[0][1]
        transcript = tokenizer.indices2entries(transcript)

        transcript = ''.join(transcript).replace('<space>', ' ').replace('@', ' ')

        total_wer += metrics.wer(reference, transcript)
        total_cer += metrics.cer(reference, transcript)

        num_tokens += float(len(reference.split()))
        num_chars += float(len(reference))

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars

    return alpha, beta, wer, cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune language model given acoustic model and dataset')

    parser.add_argument('logits_targets_file',
                        type=str,
                        help='A file containing a list of tuple containing (log_softmax_logit, target).')
    parser.add_argument('--tokenizer-file', required=True)
    parser.add_argument('--output-file', type=str, help='path to output file', default='tune-output.csv')
    parser.add_argument('--num-workers', default=cpu_count() - 1, type=int, help='Number of parallel decodes to run')

    lm_args = parser.add_argument_group("LM Options")
    lm_args.add_argument('--lm-path', default=None, type=str, help='Language model to use')
    lm_args.add_argument('--lm-trie-path')
    lm_args.add_argument('--lm-unit', choices=['word', 'char'], default='word')

    beam_args = parser.add_argument_group("Beam Decode Options",
                                          "paramsurations options for the CTC Beam Search decoder")

    beam_args.add_argument('--beam-size', default=100, type=int, help='Beam width to use')

    beam_args.add_argument('--alpha-from', default=0.0, type=float, help='Language model weight start tuning')
    beam_args.add_argument('--alpha-to', default=3.0, type=float, help='Language model weight end tuning')
    beam_args.add_argument('--beta-from',
                           default=0.0,
                           type=float,
                           help='Language model word bonus (all words) start tuning')
    beam_args.add_argument('--beta-to',
                           default=0.5,
                           type=float,
                           help='Language model word bonus (all words) end tuning')
    beam_args.add_argument('--alpha-steps', default=45, type=float, help='Number of alpha candidates for tuning')
    beam_args.add_argument('--beta-steps', default=8, type=float, help='Number of beta candidates for tuning')

    beam_args.add_argument('--cutoff-top-n',
                           default=40,
                           type=int,
                           help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                           'vocabulary will be used in beam search, default 40.')
    beam_args.add_argument('--cutoff-prob',
                           default=1.0,
                           type=float,
                           help='Cutoff probability in pruning,default 1.0, no pruning.')

    beam_args.add_argument('--lm-workers', default=1, type=int)

    args = parser.parse_args()
    tune_from_args(args)