
#include <unistd.h>
#include <stdexcept>

#include "LM.h"
#include "decoder_utils.h" 


#include <exception>

void LM::saveTrie(const std::string & path) {
    if(dictionary == nullptr) {
        throw "No trie available.";
    }

    dictionary->Write(path.c_str());
}

void LM::loadTrie(const std::string & path) {
    if (unit != LMUnit::Word)
        throw "Trie only works for word-level language model";

    VALID_CHECK_EQ(access(path.c_str(), F_OK), 0, "[KenLM] Invalid trie path");
    dictionary.reset(FstType::Read(path.c_str()));
}

LM::FstType* LM::getTrie() const {
    return dictionary.get();
}


bool LM::hasTrie() {
    return dictionary != nullptr && unit == LMUnit::Word;
}

void LM::setupTrie(const std::vector<std::string> &vocabulary) {
    if (unit != LMUnit::Word)
        throw "Trie only works for word-level language model";

    // ConstFst is immutable, so we need to use a MutableFst to create the trie,
  // and then we convert to a ConstFst for the decoder and for storing on disk.
  fst::StdVectorFst dictionary;
  // For each unigram convert to ints and put in trie
  for (const auto& word : vocabulary) {
    auto entries = split_utf8_str(word);
    try {
        auto indices = tokenizer_->mapEntriesToIndices(entries);
        indices.push_back(tokenizer_->getSpaceIndex());
        add_word_to_fst(indices, &dictionary);
    }
    catch(std::exception& e) {
        std::cout << e.what() << ". It was not possible add the word '" << word << "' to FST" << std::endl;
    }
  }

  /* Simplify FST
   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST deterministic, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&dictionary);
  std::unique_ptr<fst::StdVectorFst> new_dict(new fst::StdVectorFst);

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(dictionary, new_dict.get());

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict.get());

  // Now we convert the MutableFst to a ConstFst (Scorer::FstType) via its ctor

  std::unique_ptr<FstType> converted(new FstType(*new_dict));
  this->dictionary = std::move(converted);
}