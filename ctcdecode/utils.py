import time
from collections import Counter

from ctcdecode.fst import FST
from ctcdecode.tokenizer import CharTokenizer


def build_fst_from_words_file(words_file, token_file, output_file, max_words=None, min_count=None):
    s = time.time()
    words = []
    with open(words_file, 'r') as f:
        for line in f:
            words.append(line.strip())

    build_fst_from_words(words, token_file, output_file, max_words, min_count)

    print(f'Found {len(words)} words in {time.time() - s} seconds.')


def build_fst_from_words(words, token_file, output_file, max_words=None, min_count=None):

    counter = Counter(words)
    tokenizer = CharTokenizer(token_file, allow_unknown=True)

    print('Generating fst...')
    fst = FST.from_vocab(
        map(
            lambda x: x[0],
            filter(lambda x: True if not min_count or (min_count and x[1] >= min_count) else False,
                   counter.most_common(max_words))), tokenizer)
    fst.save(output_file)
    print(f'FST trie saved at {output_file}.')
