#include "asr_decoder/ctc_decoder.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

#define CHECK(expression)                                                   \
  do {                                                                      \
    if (!(expression)) {                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
                   #expression);                                            \
      return 1;                                                             \
    }                                                                       \
  } while (false)

namespace {

std::vector<int> encode_independent_context(
    sentencepiece::SentencePieceProcessor& processor, const std::string& text) {
  std::vector<int> ids;
  if (!processor.Encode(text, &ids).ok()) return {};
  std::vector<int> marked;
  if (!processor.Encode("▁" + text, &marked).ok()) return {};
  if (!marked.empty() && processor.IdToPiece(marked.front()) != "▁") {
    ids = std::move(marked);
  }
  return ids;
}

}  // namespace

int main() {
  const std::string model_prefix = ASR_DECODER_TEST_MODEL_PREFIX;
  const std::string trainer_options =
      "--input=" ASR_DECODER_TEST_CORPUS " --model_prefix=" + model_prefix +
      " --model_type=bpe --vocab_size=128 --hard_vocab_limit=false"
      " --bos_id=-1 --eos_id=-1 --pad_id=-1 --add_dummy_prefix=false"
      " --remove_extra_whitespaces=false";
  CHECK(sentencepiece::SentencePieceTrainer::Train(trainer_options).ok());

  const std::string model_path = model_prefix + ".model";
  sentencepiece::SentencePieceProcessor processor;
  CHECK(processor.Load(model_path).ok());
  const std::vector<std::string> contexts{"live captions", "实时字幕"};

  // Direct mode must produce the exact IDs from SentencePiece, including the
  // independent-context word-boundary behavior used by Python.
  asr_decoder::SentencePieceConfig direct_config;
  direct_config.model_path = model_path;
  const auto direct =
      asr_decoder::tokenize_sentencepiece_contexts(contexts, direct_config);
  CHECK(direct);
  CHECK(direct.context_token_ids.size() == contexts.size());
  CHECK(direct.context_token_ids[0] ==
        encode_independent_context(processor, contexts[0]));
  CHECK(direct.context_token_ids[1] ==
        encode_independent_context(processor, contexts[1]));

  std::vector<int> plain_ids;
  CHECK(processor.Encode(contexts[0], &plain_ids).ok());
  CHECK(direct.context_token_ids[0] != plain_ids);
  auto substring_config = direct_config;
  substring_config.add_word_boundary = false;
  const auto substring = asr_decoder::tokenize_sentencepiece_contexts(
      {contexts[0]}, substring_config);
  CHECK(substring);
  CHECK(substring.context_token_ids[0] == plain_ids);

  // CTC vocabularies may reserve ID 0 for blank. Exact piece mapping must
  // preserve every token while translating to the acoustic model's IDs.
  asr_decoder::SentencePieceConfig mapped_config = direct_config;
  for (int id = 0; id < processor.GetPieceSize(); ++id) {
    mapped_config.symbol_table.emplace(processor.IdToPiece(id), id + 1);
  }
  const auto mapped =
      asr_decoder::tokenize_sentencepiece_contexts(contexts, mapped_config);
  CHECK(mapped);
  CHECK(mapped.context_token_ids.size() == contexts.size());
  for (size_t index = 0; index < contexts.size(); ++index) {
    auto expected = encode_independent_context(processor, contexts[index]);
    for (int& id : expected) ++id;
    CHECK(mapped.context_token_ids[index] == expected);
  }
  for (int id : mapped.word_boundary_token_ids) CHECK(id > 0);

  // Unknown fallback is valid for context pieces, but missing word-start
  // pieces must not turn the unknown token into a synthetic word boundary.
  auto unknown_config = direct_config;
  unknown_config.symbol_table = {{"<unk>", 7}};
  const auto unknown = asr_decoder::tokenize_sentencepiece_contexts(
      {contexts[0]}, unknown_config);
  CHECK(unknown);
  CHECK(std::all_of(unknown.context_token_ids[0].begin(),
                    unknown.context_token_ids[0].end(),
                    [](int id) { return id == 7; }));
  CHECK(unknown.word_boundary_token_ids.empty());

  asr_decoder::DecoderConfig decoder_config;
  decoder_config.contexts = contexts;
  decoder_config.sentencepiece = mapped_config;
  asr_decoder::CTCDecoder decoder(decoder_config);
  CHECK(decoder.valid());
  CHECK(decoder.create_stream().valid());

  auto ambiguous_config = decoder_config;
  ambiguous_config.context_token_ids = {{1}};
  asr_decoder::CTCDecoder ambiguous_decoder(ambiguous_config);
  CHECK(!ambiguous_decoder.valid());

  auto incomplete_config = mapped_config;
  incomplete_config.symbol_table.erase(processor.IdToPiece(
      encode_independent_context(processor, contexts[0])[0]));
  incomplete_config.symbol_table.erase("<unk>");
  const auto incomplete = asr_decoder::tokenize_sentencepiece_contexts(
      {contexts[0]}, incomplete_config);
  CHECK(!incomplete);
  return 0;
}
