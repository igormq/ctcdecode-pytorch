#ifndef BINDING_H_
#define BINDING_H_
#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "LM.h"
#include "ctc_beam_search_decoder.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> beam_decoder(const at::Tensor log_probs,
                                                                        const at::Tensor seq_lengths,
                                                                        int blank_id,
                                                                        int beam_size,
                                                                        int num_processes,
                                                                        double cutoff_prob,
                                                                        int cutoff_top_n,
                                                                        LMPtr lm);
// int paddle_beam_decode(THFloatTensor *th_probs,
//                        THIntTensor *th_seq_lens,
//                        const char* labels,
//                        int vocab_size,
//                        size_t beam_size,
//                        size_t num_processes,
//                        double cutoff_prob,
//                        size_t cutoff_top_n,
//                        size_t blank_id,
//                        int log_input,
//                        THIntTensor *th_output,
//                        THIntTensor *th_timesteps,
//                        THFloatTensor *th_scores,
//                        THIntTensor *th_out_length);

// int paddle_beam_decode_lm(THFloatTensor *th_probs,
//                           THIntTensor *th_seq_lens,
//                           const char* labels,
//                           int vocab_size,
//                           size_t beam_size,
//                           size_t num_processes,
//                           double cutoff_prob,
//                           size_t cutoff_top_n,
//                           size_t blank_id,
//                           bool log_input,
//                           int *lm,
//                           THIntTensor *th_output,
//                           THIntTensor *th_timesteps,
//                           THFloatTensor *th_scores,
//                           THIntTensor *th_out_length);

// void* paddle_get_scorer(double alpha,
//                         double beta,
//                         const char* lm_path,
//                         const char* labels,
//                         int vocab_size);

// int is_character_based(void *lm);
// size_t get_max_order(void *lm);
// size_t get_dict_size(void *lm);
// void reset_params(void *lm, double alpha, double beta);

#endif // BINDING_H_
