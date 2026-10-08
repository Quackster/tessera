#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/loaders/hf_tokenizer.hpp"
#include "test_helpers.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

using tessera::Engine;
using tessera::ModelOptions;
using tessera::StatusCode;
using tessera::Tokenizer;
using tessera::testing::MakeEngineOrSkip;

// A tiny ASCII tokenizer: each letter is one byte-level token.
TEST(TokenizerTest, TinyAsciiRoundTrip) {
  Tokenizer tok({"a", "b", "c"}, {1, 1, 1}, {});
  auto ids = tok.Encode("abc");
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  const std::vector<std::uint32_t> want = {0, 1, 2};
  EXPECT_EQ(*ids, want);
  auto text = tok.Decode(*ids);
  ASSERT_TRUE(text.has_value()) << tessera::ToString(text.error());
  EXPECT_EQ(*text, "abc");
}

// A special (control) token is found by its literal string, and a
// missing one returns nullopt.
TEST(TokenizerTest, SpecialTokenIdLookup) {
  Tokenizer tok({"a", "b", "<|image_pad|>"}, {1, 1, 3}, {});
  auto id = tok.SpecialTokenId("<|image_pad|>");
  ASSERT_TRUE(id.has_value());
  EXPECT_EQ(*id, 2u);
  EXPECT_FALSE(tok.SpecialTokenId("<|nope|>").has_value());
}

// A merge rule collapses "a" "b" into the "ab" token.
TEST(TokenizerTest, MergeApplies) {
  Tokenizer tok({"a", "b", "ab"}, {1, 1, 1}, {"a b"});
  auto ids = tok.Encode("ab");
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  const std::vector<std::uint32_t> want = {2};
  EXPECT_EQ(*ids, want);
}

// Digits split per character (the Qwen pre-tokenizer), letters group.
TEST(TokenizerTest, DigitsSplitPerChar) {
  Tokenizer tok({"a", "0", "1"}, {1, 1, 1}, {});
  auto ids = tok.Encode("a01");
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  const std::vector<std::uint32_t> want = {0, 1, 2};
  EXPECT_EQ(*ids, want);
}

// Decode rejects an out-of-range id.
TEST(TokenizerTest, DecodeRejectsBadId) {
  Tokenizer tok({"a"}, {1}, {});
  const std::vector<std::uint32_t> ids = {5};
  auto text = tok.Decode(ids);
  ASSERT_FALSE(text.has_value());
  EXPECT_EQ(text.error(), StatusCode::InvalidArgument);
}

// Encode against the reference tokenizer ids (path via TESSERA_TEST_MODEL;
// never hard-coded). Decode round-trips the same text.
TEST(TokenizerTest, RealTokenizerMatchesReference) {
  const char* path = std::getenv("TESSERA_TEST_MODEL");
  if (path == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MODEL not set";
  }
  std::unique_ptr<Engine> engine;
  MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(ModelOptions{path, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());
  const Tokenizer* tok = (*model)->GetTokenizer();
  ASSERT_NE(tok, nullptr);
  struct Case {
    const char* text;
    std::vector<std::uint32_t> ids;
  };
  const std::vector<Case> cases = {
      {"Hello", {9419}},
      {"Hello, world!", {9419, 11, 1814, 0}},
      {"1 2 3", {16, 220, 17, 220, 18}},
      {"123", {16, 17, 18}},
      {"don't", {14572, 914}},
      {"def f(x):\n    return x + 1\n",
       {727, 281, 2007, 1590, 198, 262, 460, 830, 478, 220, 16, 198}},
      {"  leading", {220, 6187}},
      {"日本語のテスト", {247359, 15303, 181801}},
  };
  for (const Case& c : cases) {
    auto got = tok->Encode(c.text);
    ASSERT_TRUE(got.has_value()) << tessera::ToString(got.error());
    EXPECT_EQ(*got, c.ids) << "text: " << c.text;
    auto back = tok->Decode(*got);
    ASSERT_TRUE(back.has_value()) << tessera::ToString(back.error());
    EXPECT_EQ(*back, c.text) << "text: " << c.text;
  }
}

// A control token (type 3) is matched literally in the input and emits its
// id; the text around it is encoded normally.
TEST(TokenizerTest, SpecialTokenMatched) {
  Tokenizer tok({"<|im_start|>", "h", "i"}, {3, 1, 1}, {});
  auto ids = tok.Encode("<|im_start|>hi");
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  const std::vector<std::uint32_t> want = {0, 1, 2};
  EXPECT_EQ(*ids, want);
  auto text = tok.Decode(*ids);
  ASSERT_TRUE(text.has_value()) << tessera::ToString(text.error());
  EXPECT_EQ(*text, "<|im_start|>hi");
}

// The HuggingFace tokenizer.json parser inverts model.vocab (token -> id),
// reads the merges and marks added_tokens as control tokens.
TEST(TokenizerTest, HfTokenizerJsonParses) {
  const char* json = R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "ab": 2, "c": 3},
      "merges": ["a b"]
    },
    "added_tokens": [{"id": 4, "content": "<|endoftext|>", "special": true}]
  })";
  auto tokenizer = tessera::core::ParseHfTokenizer(json);
  ASSERT_TRUE(tokenizer.has_value()) << tessera::ToString(tokenizer.error());
  ASSERT_TRUE(tokenizer->has_value());
  const Tokenizer& tok = **tokenizer;
  EXPECT_EQ(tok.VocabSize(), 5u);
  auto ids = tok.Encode("abc");
  ASSERT_TRUE(ids.has_value()) << tessera::ToString(ids.error());
  const std::vector<std::uint32_t> want = {2, 3};
  EXPECT_EQ(*ids, want);
  auto special = tok.SpecialTokenId("<|endoftext|>");
  ASSERT_TRUE(special.has_value());
  EXPECT_EQ(*special, 4u);
}

// A non-BPE tokenizer resolves to no tokenizer.
TEST(TokenizerTest, HfTokenizerNonBpeIsEmpty) {
  auto tokenizer =
      tessera::core::ParseHfTokenizer(R"({"model": {"type": "WordPiece"}})");
  ASSERT_TRUE(tokenizer.has_value()) << tessera::ToString(tokenizer.error());
  EXPECT_FALSE(tokenizer->has_value());
}

// A malformed tokenizer.json is rejected.
TEST(TokenizerTest, HfTokenizerRejectsMalformed) {
  auto tokenizer =
      tessera::core::ParseHfTokenizer(R"({"model": {"type": "BPE"}})");
  ASSERT_FALSE(tokenizer.has_value());
  EXPECT_EQ(tokenizer.error(), StatusCode::MalformedFile);
}
