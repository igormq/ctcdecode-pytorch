/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>

#include "Tokenizer.h"
#include "LM.h"

#include <lm/state.hh>

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

struct KenLMState
{
  lm::ngram::State state;
  std::vector<int> tokens;

  KenLMState() {}
};

enum KenLMUnit
{
  Word = 0,
  Char
};

/**
 * KenLM extends LM by using the toolkit https://kheafield.com/code/kenlm/.
 */
class KenLM : public LM
{
public:
  KenLM(const std::string &path, const Tokenizer &tokenizer, KenLMUnit unit);

  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr &state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr &state) override;

  int compareState(const LMStatePtr &state1, const LMStatePtr &state2)
      const override;

  KenLMUnit unit;
  const Tokenizer *tokenizer_;

private:
  std::shared_ptr<lm::base::Model> model_;
  const lm::base::Vocabulary *vocab_;

  static KenLMState *getRawState(const LMStatePtr &state);
};

using KenLMPtr = std::shared_ptr<KenLM>;
