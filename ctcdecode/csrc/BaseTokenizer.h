#pragma once

#include <string>
#include <unordered_map>
#include <vector>


class BaseTokenizer {
  public:
    
    virtual size_t len() const = 0;

    virtual std::vector<int> entry2Idx(std::string& entry) const = 0;

    virtual std::string idx2Entry(std::vector<int>& idxs) const = 0;

    virtual int getBlankIdx() const = 0;

    virtual int getSpaceIdx() const = 0;

    virtual ~BaseTokenizer() = default;
};