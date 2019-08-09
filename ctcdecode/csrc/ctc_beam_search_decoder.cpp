#include "ctc_beam_search_decoder.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "KenLM.h"
#include "path_trie.h"


DecoderState *
decoder_init(int blank_id,
             int class_dim,
             const LMPtr lm)
{

  // assign special ids
  DecoderState *state = new DecoderState;
  state->time_step = 0;
  state->blank_id = blank_id;

  // init prefixes' root
  PathTrie *root = new PathTrie;
  root->score = root->score_ctc = root->score_lm = root->p_b = 0.0;

  if (lm != nullptr)
  {
    auto lm_state_ptr = lm->start(0);
    root->lmState = lm_state_ptr;
  }

  state->prefix_root = root;

  state->prefixes.push_back(root);

  // if (lm != nullptr && !lm->is_character_based()) {
  //   auto dict_ptr = lm->dictionary->Copy(true);
  //   root->set_dictionary(dict_ptr);
  //   auto matcher = std::make_shared<fst::SortedMatcher<PathTrie::FstType>>(*dict_ptr, fst::MATCH_INPUT);
  //   root->set_matcher(matcher);
  // }

  return state;
}

void decoder_next(const float *log_probs,
                  DecoderState *state,
                  int time_dim,
                  int class_dim,
                  double log_cutoff_prob,
                  size_t cutoff_top_n,
                  size_t beam_size,
                  const LMPtr lm,
                  double alpha,
                  double beta)
{
  // prefix search over time
  for (size_t rel_time_step = 0; rel_time_step < time_dim; ++rel_time_step, ++state->time_step)
  {
    auto *log_prob = &log_probs[rel_time_step * class_dim];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;

    if (lm != nullptr)
    {
      size_t num_prefixes = std::min(state->prefixes.size(), beam_size);
      std::sort(
          state->prefixes.begin(), state->prefixes.begin() + num_prefixes, prefix_compare);

      min_cutoff = state->prefixes[num_prefixes - 1]->score +
                   log_prob[state->blank_id] - std::max(0.0, beta);
      full_beam = (num_prefixes == beam_size);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(log_prob, class_dim, log_cutoff_prob, cutoff_top_n);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++)
    {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < state->prefixes.size() && i < beam_size; ++i)
      {
        auto prefix = state->prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff)
          break;

        // blank
        if (c == state->blank_id)
        {
          prefix->n_p_b =
              log_sum_exp(prefix->n_p_b, log_prob_c + prefix->score_ctc);
          continue;
        }

        // repeated character
        if (c == prefix->character)
          prefix->n_p_nb = log_sum_exp(
              prefix->n_p_nb, log_prob_c + prefix->p_nb);

        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, state->time_step, log_prob_c);

        if (prefix_new == nullptr)
          continue;

        float log_p = -NUM_FLT_INF;

        if (c == prefix->character && prefix->p_b > -NUM_FLT_INF)
          log_p = log_prob_c + prefix->p_b;
        else if (c != prefix->character)
          log_p = log_prob_c + prefix->score_ctc;

        prefix_new->n_p_nb =
            log_sum_exp(prefix_new->n_p_nb, log_p);

        // language model scoring
        if (lm == nullptr)
          continue;

        auto lm_out = lm->score(prefix->lmState, prefix_new->character);

        // int lmCmp = lm_->compareState(prefix_new->lmState, lm_out->first);

        // if (lmCmp != 0) {
        //   // diff state
        //   if (lmCmp > 0) //actual state is better
        //     continue;

        //   prefix_new->lmState = lm_out.first;
        //   prefix_new->score_lm = prefix->score_lm;
        //   if (lm_out.second <= 0.0)
        //     prefix_new->score_lm += lm_out.second * alpha + beta;

        // } else {
        //   // same state
        //   if (prefix_new->score_lm > prefix->score_lm + lm_out.second * alpha + beta) {
        //     // nothing to do here
        //     continue;
        //   }
        //   prefix_new->score_lm = prefix->score_lm + lm_out.second * alpha + beta;
        // }
        //   if (prefix_new->lmState.get() != nullptr) {
        //   delete prefix_new->lmState.get();
        // }

        prefix_new->lmState = lm_out.first;
        prefix_new->score_lm = prefix->score_lm;
        if (lm_out.second <= 0.0)
          prefix_new->score_lm += lm_out.second * alpha + beta;
      } // end of loop over prefix
    }   // end of loop over vocabulary

    // update log log_probs
    state->prefixes.clear();
    state->prefix_root->iterate_to_vec(state->prefixes, (lm != nullptr));

    // only preserve top beam_size prefixes
    if (state->prefixes.size() >= beam_size)
    {
      std::nth_element(state->prefixes.begin(),
                       state->prefixes.begin() + beam_size,
                       state->prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < state->prefixes.size(); ++i)
        state->prefixes[i]->remove();

      // Remove the elements from std::vector
      state->prefixes.resize(beam_size);
    }

  } // end of loop over time
}

std::vector<Output> decoder_decode(DecoderState *state,
                                   size_t beam_size,
                                   const LMPtr lm, double alpha, double beta)
{
  std::vector<PathTrie *> prefixes_copy = state->prefixes;
  std::unordered_map<const PathTrie *, float> scores;
  for (PathTrie *prefix : prefixes_copy)
    scores[prefix] = prefix->score;

  // score the last word of each prefix that doesn't end with space
  if (lm != nullptr)
  {
    for (size_t i = 0; i < beam_size && i < prefixes_copy.size(); ++i)
    {
      auto prefix = prefixes_copy[i];
      if (!prefix->is_empty())
      {
        auto lm_out = lm->finish(prefix->lmState);
        if (lm_out.second <= 0.0)
          scores[prefix] += lm_out.second * alpha + beta;
      }
    }
  }

  using namespace std::placeholders;
  size_t num_prefixes = std::min(prefixes_copy.size(), beam_size);
  std::sort(prefixes_copy.begin(), prefixes_copy.begin() + num_prefixes, std::bind(prefix_compare_external, _1, _2, scores));

  return get_beam_search_result(prefixes_copy, beam_size);
}

std::vector<Output> ctc_beam_search_decoder(
    const float *log_probs,
    int time_dim,
    int class_dim,
    int blank_id,
    size_t beam_size,
    double log_cutoff_prob,
    size_t cutoff_top_n,
    const LMPtr lm, double alpha, double beta)
{

  DecoderState *state = decoder_init(blank_id, class_dim, lm);
  decoder_next(log_probs, state, time_dim, class_dim, log_cutoff_prob, cutoff_top_n, beam_size, lm, alpha, beta);
  std::vector<Output> out = decoder_decode(state, beam_size, lm, alpha, beta);

  delete state;

  return out;
}

std::vector<std::vector<Output>>
ctc_beam_search_decoder_batch(
    const float *log_probs,
    int batch_size,
    int time_dim,
    int class_dim,
    const int *seq_lengths,
    int seq_lengths_size,
    int blank_id,
    size_t beam_size,
    size_t num_processes,
    double log_cutoff_prob,
    size_t cutoff_top_n,
    const LMPtr lm, double alpha, double beta)
{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative");
  VALID_CHECK_EQ(batch_size, seq_lengths_size, "must have one sequence length per batch element");
  // thread pool
  ThreadPool pool(num_processes);

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<Output>>> res;
  for (size_t i = 0; i < batch_size; ++i)
  {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  &log_probs[i * time_dim * class_dim],
                                  seq_lengths[i],
                                  class_dim,
                                  blank_id,
                                  beam_size,
                                  log_cutoff_prob,
                                  cutoff_top_n,
                                  lm, alpha, beta));
  }

  // // get decoding results
  std::vector<std::vector<Output>> batch_results;
  for (size_t i = 0; i < batch_size; ++i)
  {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
