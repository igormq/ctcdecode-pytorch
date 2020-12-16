
#pragma once

#include <memory>
#include <unordered_map>
#include <utility>
#include <cstring>
#include <vector>

#include "fst/fstlib.h"
#include "Tokenizer.h"


const float OOV_SCORE = -1000.0;

enum LMUnit
{
  Word = 0,
  Char
};

struct LMState
{
  std::unordered_map<int, std::shared_ptr<LMState>> children;

  template <typename T>
  std::shared_ptr<T> child(int usrIdx)
  {
    auto s = children.find(usrIdx);
    if (s == children.end())
    {
      auto state = std::make_shared<T>();
      children[usrIdx] = state;
      return state;
    }
    else
    {
      return std::static_pointer_cast<T>(s->second);
    }
  }

  /* Compare two language model states. */
  int compare(const std::shared_ptr<LMState> &state) const
  {
    LMState *inState = state.get();
    if (!state)
    {
      throw std::runtime_error("a state is null");
    }
    if (this == inState)
    {
      return 0;
    }
    else if (this < inState)
    {
      return -1;
    }
    else
    {
      return 1;
    }
  };
}

/**
 * LMStatePtr is a shared LMState* tracking LM states generated during decoding.
 */
using LMStatePtr = std::shared_ptr<LMState>;

  /**
 * LM is a thin wrapper for laguage models. We abstrct several common methods
 * here which can be shared for KenLM, ConvLM, RNNLM, etc.
 */
  class LM
  {

  public:
    LM(const Tokenizer &tokenizer, LMUnit unit) : tokenizer_(&tokenizer), unit(unit){};

    using FstType = fst::ConstFst<fst::StdArc>;
    /* Initialize or reset language model */
    virtual LMStatePtr start(bool startWithNothing) = 0;

    /**
   * Query the language model given input language model state and a specific
   * token, return a new language model state and score.
   */
    virtual std::pair<LMStatePtr, float> score(
        const LMStatePtr &state,
        const int usrTokenIdx) = 0;

    /* Query the language model and finish decoding. */
    virtual std::pair<LMStatePtr, float> finish(const LMStatePtr &state) = 0;

    /* Update LM caches (optional) given a bunch of new states generated */
    virtual void updateCache(std::vector<LMStatePtr> stateIdices) {}

    virtual ~LM() = default;

    void saveTrie(const std::string &path);
    void loadTrie(const std::string &path);

    bool hasTrie();

    FstType *getTrie() const;

  protected:
    void setupTrie(const std::vector<std::string> &vocabulary);
    // pointer to the dictionary of FST
    std::unique_ptr<FstType> dictionary;
    const Tokenizer *tokenizer_;
    LMUnit unit;
};

using LMPtr = std::shared_ptr<LM>;