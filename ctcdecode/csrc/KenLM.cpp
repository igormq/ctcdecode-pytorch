#include "KenLM.h"

#include <stdexcept>
#include <lm/model.hh>
#include "decoder_utils.h"

KenLM::KenLM(const std::string& path, const Tokenizer& tokenizer, KenLMUnit unit) {
  // Load LM
  model_.reset(lm::ngram::LoadVirtual(path.c_str()));
  if (!model_) {
    throw std::runtime_error("[KenLM] LM loading failed.");
  }
  vocab_ = &model_->BaseVocabulary();
  if (!vocab_) {
    throw std::runtime_error("[KenLM] LM vocabulary loading failed.");
  }

  unit = unit;
  tokenizer_ = &tokenizer;
}

LMStatePtr KenLM::start(bool startWithNothing) {
  auto outState = std::make_shared<KenLMState>();
  
  if (startWithNothing) {
    model_->NullContextWrite(&outState->state);
  } else {
    model_->BeginSentenceWrite(&outState->state);
  }

  return outState;
}

std::pair<LMStatePtr, float> KenLM::score(
    const LMStatePtr& state,
    const int token_index) {

  auto inState = getRawState(state);

  if (token_index != tokenizer_->getSpaceIndex()) {
    inState->tokens.push_back(token_index);
    return std::make_pair(state, -NUM_FLT_MIN);
  }

  auto entries = tokenizer_->mapIndicesToEntries(inState->tokens);
  std::string word;
  for (const auto &piece: entries) word += piece;

  auto outState = std::make_shared<KenLMState>();

  float score =
      model_->BaseScore(&inState->state, vocab_->Index(word), &outState->state);

  if (vocab_->Index(word) == 0) {
    score = OOV_SCORE;
  }

  return std::make_pair(std::move(outState), score);
}

std::pair<LMStatePtr, float> KenLM::finish(const LMStatePtr& state) {
  auto inState = getRawState(state);

  float score = 0.0;

  if (inState->tokens.size() > 0) {
    auto output = KenLM::score(state, tokenizer_->getSpaceIndex());
    inState = getRawState(output.first);
    score += output.second;
  }

  auto outState = std::make_shared<KenLMState>();
  score +=
      model_->BaseScore(&inState->state, vocab_->EndSentence(), &outState->state);

  return std::make_pair(std::move(outState), score);
}

int KenLM::compareState(const LMStatePtr& state1, const LMStatePtr& state2)
    const {
  auto inState1 = getRawState(state1);
  auto inState2 = getRawState(state2);
  if (inState1->state == inState2->state) {
    return 0;
  }
  return inState1->state.Compare(inState2->state);
}

KenLMState* KenLM::getRawState(const LMStatePtr& state) {
  return static_cast<KenLMState*>(state.get());
}