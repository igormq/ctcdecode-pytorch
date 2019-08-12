/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>

#include "LM.h"

#include "lm/enumerate_vocab.hh"
#include "lm/word_index.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"

// KenLM forward declarations
namespace lm
{
namespace base
{
class Model;
class Vocabulary;
} // namespace base
// namespace ngram
// {
// class State;
// } // namespace ngram
} // namespace lm

// Implement a callback to retrieve the dictionary of language model.
class RetrieveStrEnumerateVocab : public lm::EnumerateVocab {
public:
  RetrieveStrEnumerateVocab() {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    vocabulary.push_back(std::string(str.data(), str.length()));
  }

  std::vector<std::string> vocabulary;
};

struct KenLMState
{
  lm::ngram::State state;
  // WordIndex words[KENLM_MAX_ORDER - 1];
  //   float backoff[KENLM_MAX_ORDER - 1];
  //   unsigned char length;
  // typedef unsigned int WordIndex;
  std::vector<int> tokens;
};

/**
 * KenLM extends LM by using the toolkit https://kheafield.com/code/kenlm/.
 */
class KenLM : public LM
{
public:
  KenLM(const std::string &path, const Tokenizer &tokenizer, const std::string& trie_path = nullptr, LMUnit unit = LMUnit::Word, bool build_trie = false);

  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr &state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr &state) override;

  int compareState(const LMStatePtr &state1, const LMStatePtr &state2)
      const override;

private:
  std::shared_ptr<lm::base::Model> model_;
  const lm::base::Vocabulary *vocab_;

  static KenLMState *getRawState(const LMStatePtr &state);
};

using KenLMPtr = std::shared_ptr<KenLM>;
