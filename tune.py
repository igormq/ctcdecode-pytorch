import argparse
import logging
import os
from itertools import product
from timeit import default_timer

import tqdm
import pandas as pd
import torch
from torch.multiprocessing import Pool, cpu_count

from ctcdecode.tokenizers import CharTokenizer

logger = logging.getLogger('asr')


def tune(args):
    params_grid = list(
        product(torch.linspace(args.alpha_from, args.alpha_to, args.alpha_steps),
                torch.linspace(args.beta_from, args.beta_to, args.beta_steps)))

    print('Scheduling {} jobs for alphas=linspace({}, {}, {}), betas=linspace({}, {}, {})'.format(
        len(params_grid), args.alpha_from, args.alpha_to, args.alpha_steps, args.beta_from, args.beta_to,
        args.beta_steps))

    # start worker processes
    logger.info(f"Using {args.num_workers} processes and {args.lm_workers} for each CTCDecoder.")
    extract_start = default_timer()

    alphabet = CharTokenizer(args.vocab_path)

    p = Pool(args.num_workers, init, [
        args.logits_file, alphabet, args.lm_path, args.cutoff_top_n, args.cutoff_prob, args.beam_width, args.lm_workers
    ])

    scores = []
    best_wer = float('inf')
    with tqdm(p.imap(tune_step, params_grid), total=len(params_grid), desc='Grid search') as pbar:
        for params in pbar:
            alpha, beta, wer, cer = params
            scores.append([alpha, beta, wer, cer])

            if wer < best_wer:
                best_wer = wer
                pbar.set_postfix(alpha=alpha, beta=beta, wer=wer, cer=cer)

    logger.info(f"Finished {len(params_grid)} processes in {default_timer() - extract_start:.1f}s")

    df = pd.DataFrame(scores, columns=['alpha', 'beta', 'wer', 'cer'])
    df.to_csv(os.path.join(tune_dir, basename + '.csv'), index=False)


def init(logits_file, alphabet, lm_path, cutoff_top_n, cutoff_prob, beam_width, workers):
    global saved_outputs
    global decoder

    saved_outputs = torch.load(logits_file)
    decoder = BeamCTCDecoder(alphabet,
                             lm_path=lm_path,
                             cutoff_top_n=cutoff_top_n,
                             cutoff_prob=cutoff_prob,
                             beam_width=beam_width,
                             num_processes=workers)


def tune_step(params):
    alpha, beta = params
    global decoder
    global saved_outputs

    decoder.reset_params(alpha, beta)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    for i, (output, reference) in enumerate(saved_outputs):

        transcript = decoder.decode(torch.as_tensor(output, dtype=torch.float32).unsqueeze(0))[0][0]

        total_wer += decoder.wer(transcript, reference)
        total_cer += decoder.cer(transcript, reference)
        num_tokens += float(len(reference.split()))
        num_chars += float(len(reference))

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars

    return alpha, beta, wer, cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune language model given acoustic model and dataset')

    parser.add_argument('logits_file', type=str, help='path to the logits file')
    parser.add_argument('--num-workers', default=cpu_count() - 1, type=int, help='Number of parallel decodes to run')

    beam_args = parser.add_argument_group("Beam Decode Options",
                                          "paramsurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--lm-path', default=None, type=str, help='Language model to use')
    beam_args.add_argument('--fst-path', default=None, type=str, help='Vocab FST path')
    beam_args.add_argument('--vocab-path', required=True)
    beam_args.add_argument('--beam-width', default=10, type=int, help='Beam width to use')

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
    beam_args.add_argument('--unit', choices=['word', 'char'], default='word')

    args = parser.parse_args()
    tune(args)
