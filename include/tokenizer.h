#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>
class BPE {
private:
  std::map<std::pair<int, int>, int> merges;
  int next_token_id;
  std::map<int, std::string> vocab;

public:
  BPE() : next_token_id(256) {
    for (size_t i = 0; i < 256; i++) {
      vocab[i] = std::string(1, static_cast<char>(i));
    }
  }

  std::map<std::pair<int, int>, int>
  get_pair_counts(const std::vector<int> &tokens) {
    std::map<std::pair<int, int>, int> pair_count;
    for (size_t i = 0; i < tokens.size() - 1; i++) {
      pair_count[std::make_pair(tokens[i], tokens[i + 1])]++;
    }

    return pair_count;
  }

  std::vector<int> merge_tokens(const std::vector<int> &tokens,
                                std::pair<int, int> pair_merge,
                                size_t new_token_id) {
    std::vector<int> new_tokens;
    size_t i = 0;
    while (i < tokens.size()) {
      if (i < tokens.size() - 1 && pair_merge.first == tokens[i] &&
          pair_merge.second == tokens[i + 1]) {
        new_tokens.push_back(new_token_id);
        i += 2;
      } else {
        new_tokens.push_back(tokens[i]);
        i++;
      }
    }
    return new_tokens;
  }

  void train(std::string &text, int num_merges = 20) {
    std::vector<int> tokens;
    for (unsigned char c : text) {
      tokens.push_back(static_cast<int>(c));
    }

    for (int step = 0; step < num_merges; step++) {
      auto pair_counts = get_pair_counts(tokens);
      if (pair_counts.empty()) {
        break;
      }

      auto most_frequent = std::max_element(
          pair_counts.begin(), pair_counts.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });
      std::pair<int, int> pair_to_merge = most_frequent->first;
      int new_token = next_token_id++;

      vocab[new_token] =
          vocab[pair_to_merge.first] + vocab[pair_to_merge.second];

      merges[pair_to_merge] = new_token;

      tokens = merge_tokens(tokens, pair_to_merge, new_token);
    }
  }

  std::vector<int> encode(const std::string &text) {
    std::vector<int> tokens;
    for (unsigned char c : text) {
      tokens.push_back(static_cast<int>(c));
    }

    for (const auto &merge : merges) {
      tokens = merge_tokens(tokens, merge.first, merge.second);
    }
    return tokens;
  }

  std::string decode(const std::vector<int> &tokens) {
    std::string result;
    for (int token : tokens) {
      if (vocab.find(token) != vocab.end()) {
        result += vocab[token];
      }
    }
    return result;
  }
};

#endif
