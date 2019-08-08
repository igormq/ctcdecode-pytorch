/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Tokenizer.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Utils.h"

Tokenizer::Tokenizer(std::istream& stream) {
  createFromStream(stream);
}

Tokenizer::Tokenizer(const std::string& filename) {
  if (!fileExists(filename)) {
    throw std::invalid_argument(
        "Tokenizer file '" + filename + "' does not exist.");
  }
  std::ifstream stream(filename);
  createFromStream(stream);
}

void Tokenizer::createFromStream(std::istream& stream) {
  if (!stream) {
    throw std::runtime_error("Unable to open dictionary input stream.");
  }
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    auto tkns = splitOnWhitespace(line, true);
    auto idx = idx2entry_.size();
    // All entries on the same line map to the same index
    for (const auto& tkn : tkns) {
      addEntry(tkn, idx);
    }
  }
  if (!isContiguous()) {
    throw std::runtime_error("Invalid dictionary format - not contiguous");
  }
}

void Tokenizer::addEntry(const std::string& entry, int idx) {
  if (entry2idx_.find(entry) != entry2idx_.end()) {
    throw std::invalid_argument(
        "Duplicate entry name in dictionary '" + entry + "'");
  }
  entry2idx_[entry] = idx;
  if (idx2entry_.find(idx) == idx2entry_.end()) {
    idx2entry_[idx] = entry;
  }
}

void Tokenizer::addEntry(const std::string& entry) {
  // Check if the entry already exists in the dictionary
  if (entry2idx_.find(entry) != entry2idx_.end()) {
    throw std::invalid_argument(
        "Duplicate entry in dictionary '" + entry + "'");
  }
  int idx = idx2entry_.size();
  // Find first available index.
  while (idx2entry_.find(idx) != idx2entry_.end()) {
    ++idx;
  }
  addEntry(entry, idx);
}

std::string Tokenizer::getEntry(int idx) const {
  auto iter = idx2entry_.find(idx);
  if (iter == idx2entry_.end()) {
    throw std::invalid_argument(
        "Unknown index in dictionary '" + std::to_string(idx) + "'");
  }
  return iter->second;
}

int Tokenizer::getBlankIndex() const {
  if (blankIndex_ < 0){
      throw std::invalid_argument(
          "Blank index not set");
    }
  return blankIndex_;
}

int Tokenizer::getSpaceIndex() const {
  if (spaceIndex_ < 0){
      throw std::invalid_argument(
          "Space index not set");
    }
  return spaceIndex_;
}

int Tokenizer::getDefaultIndex() const {
  return defaultIndex_;
}

void Tokenizer::setBlankIndex(int idx) {
  blankIndex_ = idx;
}

void Tokenizer::setSpaceIndex(int idx) {
  spaceIndex_ = idx;
}

void Tokenizer::setDefaultIndex(int idx) {
  defaultIndex_ = idx;
}

int Tokenizer::getIndex(const std::string& entry) const {
  auto iter = entry2idx_.find(entry);
  if (iter == entry2idx_.end()) {
    if (defaultIndex_ < 0) {
      throw std::invalid_argument(
          "Unknown entry in dictionary: '" + entry + "'");
    } else {
      std::cerr << "Skipping unknown entry: '" << entry << "'" << std::endl;
      return defaultIndex_;
    }
  }
  return iter->second;
}

bool Tokenizer::contains(const std::string& entry) const {
  auto iter = entry2idx_.find(entry);
  if (iter == entry2idx_.end()) {
    return false;
  }
  return true;
}

size_t Tokenizer::entrySize() const {
  return entry2idx_.size();
}

bool Tokenizer::isContiguous() const {
  for (size_t i = 0; i < indexSize(); ++i) {
    if (idx2entry_.find(i) == idx2entry_.end()) {
      return false;
    }
  }
  for (const auto& tknidx : entry2idx_) {
    if (idx2entry_.find(tknidx.second) == idx2entry_.end()) {
      return false;
    }
  }
  return true;
}

std::vector<int> Tokenizer::mapEntriesToIndices(
    const std::vector<std::string>& entries) const {
  std::vector<int> indices;
  indices.reserve(entries.size());
  for (const auto& tkn : entries) {
    indices.emplace_back(getIndex(tkn));
  }
  return indices;
}

std::vector<std::string> Tokenizer::mapIndicesToEntries(
    const std::vector<int>& indices) const {
  std::vector<std::string> entries;
  entries.reserve(indices.size());
  for (const auto& idx : indices) {
    entries.emplace_back(getEntry(idx));
  }
  return entries;
}

size_t Tokenizer::indexSize() const {
  return idx2entry_.size();
}