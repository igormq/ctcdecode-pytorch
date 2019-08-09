#include "KenLM.h"

#include <unistd.h>

#include <stdexcept>
#include <lm/model.hh>

#include "decoder_utils.h"

KenLM::KenLM(const std::string &path, const Tokenizer &tokenizer, KenLMUnit unit)
{
  const char *filename = path.c_str();
  VALID_CHECK_EQ(access(filename, F_OK), 0, "[KenLM] Invalid language model path");

  // Load LM
  model_.reset(lm::ngram::LoadVirtual(filename));
  if (!model_)
  {
    throw std::runtime_error("[KenLM] LM loading failed.");
  }

  vocab_ = &model_->BaseVocabulary();
  if (!vocab_)
  {
    throw std::runtime_error("[KenLM] LM vocabulary loading failed.");
  }

  unit = unit;
  tokenizer_ = &tokenizer;
}

LMStatePtr KenLM::start(bool startWithNothing)
{
  auto outState = std::make_shared<KenLMState>();

  if (startWithNothing)
  {
    model_->NullContextWrite(&outState->state);
  }
  else
  {
    model_->BeginSentenceWrite(&outState->state);
  }

  return outState;
}

std::pair<LMStatePtr, float> KenLM::score(
    const LMStatePtr &state,
    const int token_index)
{
  float score;
  auto inState = getRawState(state);
  auto outState = std::make_shared<KenLMState>();

  if (unit == KenLMUnit::Word && token_index != tokenizer_->getSpaceIndex())
  {
    *outState = *inState;
    outState->tokens.push_back(token_index);
    return std::make_pair(std::move(outState), 1); // return an invalid prob, then decoder will take care
  }

  std::string entry;
  if (unit == KenLMUnit::Word)
  {
    auto entries = tokenizer_->mapIndicesToEntries(inState->tokens);
    for (const auto &piece : entries)
      entry += piece;
  }
  else
  {
    entry = tokenizer_->getEntry(token_index);
  }

  auto lm_token_index = vocab_->Index(entry);

  if (lm_token_index == 0)
  {
    score = OOV_SCORE;
  }
  else
  {
    // Some bug here
    score =
      model_->BaseScore(&inState->state, lm_token_index, &outState->state) / NUM_FLT_LOGE;
  }


  return std::make_pair(std::move(outState), score);
}

std::pair<LMStatePtr, float> KenLM::finish(const LMStatePtr &state)
{
  auto inState = getRawState(state);
  auto outState = std::make_shared<KenLMState>();
  float score = 0.0;

  if (unit == KenLMUnit::Word && inState->tokens.size() > 0)
  {
    auto output = KenLM::score(state, tokenizer_->getSpaceIndex());
    auto inState = getRawState(output.first);
    score += output.second;
  }

  score +=
      model_->BaseScore(&inState->state, vocab_->EndSentence(), &outState->state);

  return std::make_pair(std::move(outState), score / NUM_FLT_LOGE);
}

int KenLM::compareState(const LMStatePtr &state1, const LMStatePtr &state2)
    const
{
  auto inState1 = getRawState(state1);
  auto inState2 = getRawState(state2);
  if (inState1->state == inState2->state)
  {
    return 0;
  }
  return inState1->state.Compare(inState2->state);
}

KenLMState *KenLM::getRawState(const LMStatePtr &state)
{
  return static_cast<KenLMState *>(state.get());
}