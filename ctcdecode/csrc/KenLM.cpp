#include "KenLM.h"

#include <unistd.h>

#include <stdexcept>
#include <lm/model.hh>

#include "decoder_utils.h"

static const int32_t MAGIC = 'TRIE';
static const int32_t FILE_VERSION = 4;

KenLM::KenLM(const std::string &path, const Tokenizer &tokenizer, const std::string& trie_path, LMUnit unit, bool build_trie) : LM(tokenizer, unit)
{
  bool has_trie = trie_path.size() && access(trie_path.c_str(), R_OK) == 0;
  lm::ngram::Config config;

  const char *filename = path.c_str();
  VALID_CHECK_EQ(access(filename, F_OK), 0, "[KenLM] Invalid language model path");

  if (!has_trie && build_trie) { // no trie was specified, build it now
    RetrieveStrEnumerateVocab enumerate;
    config.enumerate_vocab = &enumerate;
  }

  // Load LM
  model_.reset(lm::ngram::LoadVirtual(filename, config));
  if (!model_)
  {
    throw std::runtime_error("[KenLM] LM loading failed.");
  }

  vocab_ = &model_->BaseVocabulary();
  if (!vocab_)
  {
    throw std::runtime_error("[KenLM] LM vocabulary loading failed.");
  }

  if (has_trie) {
    loadTrie(trie_path);
  }
  else if (build_trie) {
    setupTrie(static_cast<RetrieveStrEnumerateVocab*>(config.enumerate_vocab)->vocabulary);
  }
}

LMStatePtr KenLM::start(bool startWithNothing)
{
  auto outState = std::make_shared<KenLMState>();

  if (startWithNothing)
  {
    model_->NullContextWrite(outState->ken_state());
  }
  else
  {
    model_->BeginSentenceWrite(outState->ken_state());
  }

  return outState;
}

std::pair<LMStatePtr, float> KenLM::score(
    const LMStatePtr &state,
    const int token_index)
{
  std::string entry = "";
  float score;
  auto inState = std::static_pointer_cast<KenLMState>(state);
  auto outState = inState->child<KenLMState>(token_index);

  if ( unit == LMUnit::Word && token_index != tokenizer_->getSpaceIndex())
  {
    *outState = *inState; // copy struct
    outState->tokens.push_back(token_index);
    return std::make_pair(std::move(outState), 1); // return an invalid prob, then decoder will take care
  }
  
  if (unit == LMUnit::Word)
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
      model_->BaseScore(inState->ken_state(), lm_token_index, outState->ken_state()) / NUM_FLT_LOGE;
  }

  return std::make_pair(std::move(outState), score);
}

std::pair<LMStatePtr, float> KenLM::finish(const LMStatePtr &state)
{
  auto inState = std::static_pointer_cast<KenLMState>(state);
  auto outState = inState->child<KenLMState>(-1);
  float score = 0.0;

  if (unit == LMUnit::Word && inState->tokens.size() > 0)
  {
    auto output = KenLM::score(state, tokenizer_->getSpaceIndex());
    inState = std::static_pointer_cast<KenLMState>(output.first);
    score += output.second;
  }

  score +=
      model_->BaseScore(inState->ken_state(), vocab_->EndSentence(), outState->ken_state());

  return std::make_pair(std::move(outState), score / NUM_FLT_LOGE);
}
