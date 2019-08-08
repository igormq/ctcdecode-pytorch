#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
    public:
        
        // Creates an empty dictionary
        Tokenizer() {}

        explicit Tokenizer(std::istream& stream);

        explicit Tokenizer(const std::string& filename);

        size_t len() const;

        int getBlankIndex() const;

        int getSpaceIndex() const;

        int getDefaultIndex() const;

        void setBlankIndex(int idx);

        void setSpaceIndex(int idx);

        void setDefaultIndex(int idx);

        void addEntry(const std::string& entry, int idx);

        void addEntry(const std::string& entry);

        int getIndex(const std::string& entry) const;

        std::string getEntry(int idx) const;

        bool contains(const std::string& entry) const;

        // checks if all the indices are contiguous
        bool isContiguous() const;

        std::vector<int> mapEntriesToIndices(
            const std::vector<std::string>& entries) const;

        std::vector<std::string> mapIndicesToEntries(
            const std::vector<int>& indices) const;

        size_t entrySize() const; 

        size_t indexSize() const;


    private:
        // Creates a dictionary from an input stream
        void createFromStream(std::istream& stream);

        std::unordered_map<std::string, int> entry2idx_;
        std::unordered_map<int, std::string> idx2entry_;
        int defaultIndex_ = -1;
        int spaceIndex_ = -1;
        int blankIndex_ = -1;
};