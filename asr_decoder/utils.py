# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re
import warnings


def log_add(*args) -> float:
    """
    Stable log add
    """
    if all(a == -float("inf") for a in args):
        return -float("inf")
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def tokenize_by_bpe_model(sp, txt, upper=True, *, add_word_boundary=True):
    """Tokenize mixed CJK/BPE text into the model's symbol pieces.

    SentencePiece uses the visible ``▁`` marker to distinguish a word start
    from an in-word continuation.  Context phrases are normally independent
    words, so adding that marker to a phrase that does not already start with
    whitespace prevents a short English hotword such as ``andy`` from being
    matched inside the ordinary sequence ``and they``.
    """
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r"([\u4e00-\u9fff])")
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper() if upper else txt)
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            pieces = sp.encode_as_pieces(ch_or_w)
            if add_word_boundary and ch_or_w and not ch_or_w[0].isspace() and not ch_or_w.startswith("▁"):
                marked = sp.encode_as_pieces("▁" + ch_or_w)
                if not (marked and marked[0] == "▁"):
                    pieces = marked
            tokens.extend(pieces)
    return tokens


def tokenize(contexts: list[str], symbol_table: dict[str, int], bpe_model=None, *, add_word_boundary=True):
    """
    Tokenize and convert contexts into token ids.
    """
    sp = None
    if bpe_model is not None:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    context_list = []
    for context in contexts:
        context = context.strip()
        if sp is not None:
            tokens = tokenize_by_bpe_model(sp, context, add_word_boundary=add_word_boundary)
        else:
            tokens = list(context.replace(" ", "▁"))

        labels = []
        missing = []
        for ch in tokens:
            if ch in symbol_table:
                labels.append(symbol_table[ch])
            elif "<unk>" in symbol_table:
                labels.append(symbol_table["<unk>"])
            else:
                missing.append(ch)
        if missing:
            warnings.warn(
                f"Skipping context {context!r}: tokens are missing from the symbol table: {missing}",
                stacklevel=2,
            )
            continue
        if labels:
            context_list.append(labels)
    return context_list


def tokenize_sentencepiece_contexts(
    tokenizer,
    contexts: list[str],
    *,
    add_word_boundary: bool = True,
) -> list[list[int]]:
    """Encode independent context phrases with a SentencePiece tokenizer.

    ``SentencePieceProcessor.encode("andy")`` may return the in-word pieces
    ``["and", "y"]``.  Prefixing ``▁`` makes the first piece a word-start
    piece (for example ``["▁and", "y"]``), which is the safe representation
    for an independent hotword.  If a model has no combined word-start piece
    and would emit only a standalone ``▁``, the raw encoding is retained. Set
    ``add_word_boundary=False`` only when the caller intentionally wants
    substring matching.
    """
    output = []
    for context in contexts:
        text = str(context).strip()
        raw_ids = list(tokenizer.encode(text, out_type=int))
        if not add_word_boundary or not text or text.startswith("▁"):
            output.append(raw_ids)
            continue
        marked_ids = list(tokenizer.encode("▁" + text, out_type=int))
        # Some models represent a word start as a standalone ``▁`` token and
        # have no combined word-start piece for this phrase.  In that case a
        # raw encoding is the only representation the acoustic model can
        # produce for the word, so do not manufacture an unreachable prefix.
        if marked_ids and hasattr(tokenizer, "id_to_piece") and tokenizer.id_to_piece(marked_ids[0]) == "▁":
            output.append(raw_ids)
        else:
            output.append(marked_ids)
    return output
