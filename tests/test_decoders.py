"""Test decoders"""
import os
import pickle

import numpy as np
import pytest
import torch

from ctcdecode import decoders
from ctcdecode.lm import KenLM, KenLMUnit
from ctcdecode.tokenizer import Tokenizer

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@pytest.fixture
def data():
    vocab_list = ["<blank>", "\'", ' ', 'a', 'b', 'c', 'd']
    beam_size = 20
    log_probs_seq1 = torch.as_tensor(
        [[0.1649, 0.0639, 0.2112, 0.2732, 0.0687, 0.0361, 0.1818],
        [0.0689, 0.0331, 0.2287, 0.2439, 0.0970, 0.3190, 0.0095],
        [0.0812, 0.2181, 0.1999, 0.1825, 0.0850, 0.1490, 0.0842],
        [0.0977, 0.1209, 0.1916, 0.0147, 0.2805, 0.2425, 0.0521],
        [0.0195, 0.1333, 0.0055, 0.0030, 0.2175, 0.2080, 0.4132],
        [0.0146, 0.1647, 0.1981, 0.1907, 0.1896, 0.1986, 0.0438]]).log()
    log_probs_seq2 = torch.as_tensor(
        [[0.1090, 0.0803, 0.2267, 0.0580, 0.3681, 0.1131, 0.0447],
        [0.2064, 0.0974, 0.1296, 0.0944, 0.2189, 0.1511, 0.1022],
        [0.0165, 0.4503, 0.0909, 0.1533, 0.0794, 0.0865, 0.1230],
        [0.1348, 0.0251, 0.2208, 0.1966, 0.1191, 0.0782, 0.2254],
        [0.0414, 0.1793, 0.0607, 0.4115, 0.1172, 0.1188, 0.0711],
        [0.2230, 0.1588, 0.1236, 0.2338, 0.2051, 0.0028, 0.0529]]).log()
    greedy_result = ["ac'bdc", "b'da"]
    beam_search_result = ['acdc', "b'a"]

    return vocab_list, beam_size, log_probs_seq1, log_probs_seq2, greedy_result, beam_search_result


def test_greedy_decoder(data):
    vocab_list, _, log_probs_seq1, log_probs_seq2, greedy_result, _ = data
    _, bst_result, _ = decoders.ctc_greedy_decoder(log_probs_seq1)
    assert ''.join(vocab_list[i] for i in bst_result) == greedy_result[0]

    _, bst_result, _ = decoders.ctc_greedy_decoder(log_probs_seq2)
    assert ''.join(vocab_list[i] for i in bst_result) == greedy_result[1]


def test_beam_search_decoder(data):
    vocab_list, beam_size, log_probs_seq1, log_probs_seq2, _, beam_search_result = data
    beam_result = decoders.ctc_beam_search_decoder(log_probs_seq1, beam_size=beam_size)
    assert ''.join([vocab_list[s] for s in beam_result[0][1]]) == beam_search_result[0]

    beam_result = decoders.ctc_beam_search_decoder(log_probs_seq2, beam_size=beam_size)
    assert ''.join([vocab_list[s] for s in beam_result[0][1]]) == beam_search_result[1]

def test_ctc_beam_search_decoder_tf():
    log_input = torch.tensor([
    [0, 0.6, 0, 0.4, 0, 0],
    [0, 0.5, 0, 0.5, 0, 0],
    [0, 0.4, 0, 0.6, 0, 0],
    [0, 0.4, 0, 0.6, 0, 0],
    [0, 0.4, 0, 0.6, 0, 0]
], dtype=torch.float32).log()

    beam_results = decoders.ctc_beam_search_decoder(log_input, beam_size=30)

    assert beam_results[0][1] == (1, 3)
    assert beam_results[1][1] == (1, 3, 1)
    assert beam_results[2][1] == (3, 1, 3)


def test_beam_search_decoder_batch(data):
    vocab_list, beam_size, log_probs_seq1, log_probs_seq2, _, beam_search_result = data
    beam_results = decoders.ctc_beam_search_decoder_batch(
        probs_seq_list=[log_probs_seq1, log_probs_seq2], beam_size=beam_size, num_processes=24)
    assert ''.join([vocab_list[s] for s in beam_results[0][0][1]]) == beam_search_result[0]
    assert ''.join([vocab_list[s] for s in beam_results[1][0][1]]) == beam_search_result[1]


def test_ctc_decoder_beam_search_different_blank_idx():
    input_log_prob_matrix_0 = torch.tensor(
        [
            [0.173908, 0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352],
            [0.230517, 0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581],
            [0.238763, 0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289],
            [0.20655, 0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803],
            [0.129878, 0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297],
            # Random entry added in at time=5
            [0.160671, 0.155251, 0.164444, 0.173517, 0.176138, 0.169979]
        ],
        dtype=torch.float32).log()


    results = decoders.ctc_beam_search_decoder(input_log_prob_matrix_0, blank=0, beam_size=2)

    assert len(results[0][1]) == 2
    assert len(results[1][1]) == 3
    assert np.alltrue(results[0][1] == (2, 1))
    assert np.alltrue(results[1][1] == (2, 1, 4))
    assert np.allclose(4.73437, results[0][0])
    # tf results: 7.0157223
    assert np.allclose(5.318605, results[1][0])
    # tf results: 7.8031697


def test_real_ctc_decode():
    data = np.genfromtxt(os.path.join(data_dir, "rnn_output.csv"), delimiter=';')[:, :-1]
    inputs = torch.as_tensor(data).log_softmax(1)

    tokenizer = Tokenizer(os.path.join(data_dir, 'labels.txt'))
    tokenizer.blank_index = tokenizer.get_index('<blank>')
    tokenizer.space_index = tokenizer.get_index('<space>')

    # greedy using beam
    result = decoders.ctc_greedy_decoder(inputs, blank=tokenizer.blank_index)
    txt_result = ''.join(tokenizer.idxs2entries(result[1].tolist())).replace('<space>', ' ')

    assert "the fak friend of the fomly hae tC" == txt_result

    # default beam decoding
    result = decoders.ctc_beam_search_decoder(inputs, blank=tokenizer.blank_index, beam_size=25)

    txt_result = ''.join(tokenizer.idxs2entries(result[0][1]))
    # assert "the fak friend of the fomcly hae tC" == txt_result

    # lm-based decoding
    scorer = KenLM(os.path.join(data_dir, 'bigram.arpa'), tokenizer, unit=KenLMUnit.Word)
    result = decoders.ctc_beam_search_decoder(
        inputs, lm_scorer=scorer, blank=tokenizer.blank_index, beam_size=25, alpha=2, beta=0)
    txt_result = ''.join(tokenizer.idxs2entries(result[0][1])).replace('<space>', ' ')
    assert "the fake friend of the family, lie th" == txt_result

    # lm-based decoding with trie
    # scorer = KenLMScorer(os.path.join(data_dir, 'bigram.arpa'), tokenizer, trie_path=os.path.join(data_dir, 'trie.fst'), alpha=2.0, beta=0, unit='word')
    # result = decoders.ctc_beam_search_decoder(
    #     inputs, lm_scorer=scorer, blank=tokenizer.blank_index, beam_size=25)
    # txt_result = ''.join(tokenizer.idxs2entries(result[0][1]))
    # assert "the fake friend of the family, like the" == txt_result


def test_real_ctc_decode2():
    with open(os.path.join(data_dir, 'ctc-test.pickle'), 'rb') as f:
        seq, label = pickle.load(f, encoding='bytes')

    seq = torch.as_tensor(seq).squeeze().log_softmax(1)

    tokenizer = CharTokenizer(os.path.join(data_dir, 'toy-data-vocab.txt'))
    beam_width = 16

    result = decoders.ctc_beam_search_decoder(seq, blank=tokenizer.blank_index, beam_size=beam_width)

    txt_result = ''.join(tokenizer.idxs2entries(result[0][1]))

    assert txt_result == 'then seconds'
    assert np.allclose(1.1842575, result[0][0], atol=1e-3)


    # lm-based decoding
    scorer = KenLMScorer(os.path.join(data_dir, 'ctc-test-lm.binary'), tokenizer, alpha=2.0, beta=0.5, unit='word')
    result = decoders.ctc_beam_search_decoder(
        seq, lm_scorer=scorer, blank=tokenizer.blank_index, beam_size=beam_width)
    txt_result = ''.join(tokenizer.idxs2entries(result[0][1]))

    assert txt_result == label
    # assert np.allclose(4.619581, result[0][0], atol=1e-3)
    # assert np.allclose(4.0845, result[0][0], atol=1e-3)
