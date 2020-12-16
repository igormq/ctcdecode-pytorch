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
#include <lm/model.hh>


// Implement a callback to retrieve the dictionary of language model.
class RetrieveStrEnumerateVocab : public lm::EnumerateVocab {
public:
  RetrieveStrEnumerateVocab() {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    vocabulary.push_back(std::string(str.data(), str.length()));
  }

  std::vector<std::string> vocabulary;
};

struct KenLMState : LMState {
  lm::ngram::State ken_state_;
  std::vector<int> tokens;
  lm::ngram::State* ken_state() {
    return &ken_state_;
  }
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

private:
  std::shared_ptr<lm::base::Model> model_;
  const lm::base::Vocabulary *vocab_;

};

using KenLMPtr = std::shared_ptr<KenLM>;
