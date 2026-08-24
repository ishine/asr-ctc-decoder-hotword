#include "asr_decoder/ctc_decoder.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <unordered_set>

#include <sentencepiece_processor.h>

namespace asr_decoder {
namespace {

std::string trim_ascii_whitespace(const std::string& text) {
  auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char ch) {
    return std::isspace(ch);
  });
  auto end = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char ch) {
               return std::isspace(ch);
             }).base();
  return begin < end ? std::string(begin, end) : std::string{};
}

bool starts_with_word_boundary(const std::string& piece) {
  static const std::string boundary = "▁";
  return piece.compare(0, boundary.size(), boundary) == 0;
}

bool encode_context_pieces(
    const sentencepiece::SentencePieceProcessor& processor,
    const std::string& text, bool add_word_boundary,
    std::vector<std::string>& pieces, std::string& error) {
  const auto encode_status = processor.Encode(text, &pieces);
  if (!encode_status.ok()) {
    error =
        "failed to encode SentencePiece context: " + encode_status.ToString();
    return false;
  }
  if (!add_word_boundary || starts_with_word_boundary(text)) return true;

  std::vector<std::string> marked_pieces;
  const auto marked_status = processor.Encode("▁" + text, &marked_pieces);
  if (!marked_status.ok()) {
    error = "failed to encode SentencePiece word boundary: " +
            marked_status.ToString();
    return false;
  }
  // A standalone boundary token is not part of the acoustic realization of
  // the phrase. In that case the ordinary encoding is the reachable one.
  if (!marked_pieces.empty() && marked_pieces.front() != "▁") {
    pieces = std::move(marked_pieces);
  }
  return true;
}

std::optional<int> map_piece_id(
    const sentencepiece::SentencePieceProcessor& processor,
    const std::unordered_map<std::string, int>& symbol_table,
    const std::string& piece, bool allow_unknown) {
  if (symbol_table.empty()) return processor.PieceToId(piece);
  const auto found = symbol_table.find(piece);
  if (found != symbol_table.end()) return found->second;
  if (!allow_unknown) return std::nullopt;
  const auto unknown = symbol_table.find("<unk>");
  if (unknown != symbol_table.end()) return unknown->second;
  return std::nullopt;
}

}  // namespace

SentencePieceTokenizationResult tokenize_sentencepiece_contexts(
    const std::vector<std::string>& contexts,
    const SentencePieceConfig& config) {
  SentencePieceTokenizationResult result;
  sentencepiece::SentencePieceProcessor processor;
  const auto load_status = processor.Load(config.model_path);
  if (!load_status.ok()) {
    result.error =
        "failed to load SentencePiece model: " + load_status.ToString();
    return result;
  }

  for (const std::string& context : contexts) {
    const std::string text = trim_ascii_whitespace(context);
    if (text.empty()) continue;

    std::vector<std::string> pieces;
    if (!encode_context_pieces(processor, text, config.add_word_boundary,
                               pieces, result.error))
      return result;

    std::vector<int> ids;
    ids.reserve(pieces.size());
    for (const std::string& piece : pieces) {
      const auto id = map_piece_id(processor, config.symbol_table, piece, true);
      if (!id) {
        std::ostringstream message;
        message << "SentencePiece piece '" << piece
                << "' is missing from the symbol table for context '" << text
                << "'";
        result.error = message.str();
        return result;
      }
      ids.push_back(*id);
    }
    if (!ids.empty()) result.context_token_ids.push_back(std::move(ids));
  }

  std::unordered_set<int> boundary_ids;
  for (int id = 0; id < processor.GetPieceSize(); ++id) {
    const std::string piece = processor.IdToPiece(id);
    if (!starts_with_word_boundary(piece)) continue;
    const auto mapped =
        map_piece_id(processor, config.symbol_table, piece, false);
    if (mapped) boundary_ids.insert(*mapped);
  }
  result.word_boundary_token_ids.assign(boundary_ids.begin(),
                                        boundary_ids.end());
  std::sort(result.word_boundary_token_ids.begin(),
            result.word_boundary_token_ids.end());
  return result;
}

}  // namespace asr_decoder
