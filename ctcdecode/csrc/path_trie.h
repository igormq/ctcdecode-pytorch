#ifndef PATH_TRIE_H
#define PATH_TRIE_H

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "LM.h"

/* Trie tree for prefix storing and manipulating, with a dictionary in
 * finite-state transducer for spelling correction.
 */
class PathTrie {
public:
  using FstType = fst::ConstFst<fst::StdArc>;

  PathTrie();
  ~PathTrie();

  // get new prefix after appending new char
  PathTrie* get_path_trie(int new_char, int new_timestep, float log_prob_c, bool reset = true);

  // get the prefix in index from root to current node
  PathTrie* get_path_vec(std::vector<int>& output, std::vector<int>& timesteps);

  // get the prefix in index from some stop node to current nodel
  PathTrie* get_path_vec(std::vector<int>& output,
                         std::vector<int>& timesteps,
                         int stop,
                         size_t max_steps = std::numeric_limits<size_t>::max());

  // update log probs
  void iterate_to_vec(std::vector<PathTrie*>& output, bool has_lm);

  // set dictionary for FST
  void set_dictionary(FstType* dictionary);

  void set_matcher(std::shared_ptr<fst::SortedMatcher<FstType>>);

  bool is_empty() { return ROOT_ == character; }

  // remove current path from root
  void remove();

  float p_b;
  float p_nb;
  float n_p_b;
  float n_p_nb;
  float log_prob_c;
  float score;
  float score_ctc;
  float score_lm;
  int character;
  int timestep;
  PathTrie* parent;
  LMStatePtr lmState;

private:
  int ROOT_;
  bool exists_;
  bool has_dictionary_;

  std::vector<std::pair<int, PathTrie*>> children_;

  // pointer to dictionary of FST
  FstType* dictionary_;
  FstType::StateId dictionary_state_;
  // true if finding ars in FST
  std::shared_ptr<fst::SortedMatcher<FstType>> matcher_;
};

#endif  // PATH_TRIE_H
