import pytest

from asr_decoder import ContextPolicy
from asr_decoder.context_graph import ContextGraph
from asr_decoder.prefix_score import ContextScoreState
from asr_decoder.utils import tokenize, tokenize_by_bpe_model, tokenize_sentencepiece_contexts

SYMBOLS = {"<blank>": 0, "a": 1, "b": 2, "c": 3}


def test_named_policies_hide_raw_scoring_from_application_users():
    assert ContextPolicy.conservative().completion_bonus < ContextPolicy.balanced().completion_bonus
    assert ContextPolicy.balanced().completion_bonus < ContextPolicy.aggressive().completion_bonus


def test_completed_phrase_emits_one_phrase_level_match():
    graph = ContextGraph(["ab"], SYMBOLS, policy=ContextPolicy(completion_bonus=4.0))
    state = graph.forward_one_step(graph.root, SYMBOLS["a"])
    assert tuple(graph.completed_matches(state)) == ()

    state = graph.forward_one_step(state, SYMBOLS["b"])
    assert tuple(graph.completed_matches(state)) == ((2, 4.0),)


def test_failure_outputs_report_overlapping_suffix_matches():
    graph = ContextGraph.from_token_ids(
        [[1, 2], [2]],
        policy=ContextPolicy(completion_bonus=4.0),
    )
    state = graph.forward_one_step(graph.root, 1)
    state = graph.forward_one_step(state, 2)

    assert tuple(graph.completed_matches(state)) == ((2, 4.0), (1, 4.0))


def test_existing_trie_node_can_become_context_end():
    graph = ContextGraph(["abc", "ab"], SYMBOLS)
    state = graph.forward_one_step(graph.root, SYMBOLS["a"])
    state = graph.forward_one_step(state, SYMBOLS["b"])

    assert tuple(graph.completed_matches(state))
    assert graph.num_nodes == 4


def test_context_with_missing_symbol_is_skipped_instead_of_truncated():
    with pytest.warns(UserWarning, match="missing from the symbol table"):
        contexts = tokenize(["az"], SYMBOLS)
    assert contexts == []


class _FakeSentencePiece:
    def encode_as_pieces(self, text):
        return text.split()


class _FakeSentencePieceIds:
    def encode(self, text, out_type=int):
        return [ord(piece) for piece in text]


def test_sentencepiece_context_tokenization_marks_independent_word_start():
    assert tokenize_by_bpe_model(_FakeSentencePiece(), "andy") == ["▁ANDY"]
    assert tokenize_by_bpe_model(_FakeSentencePiece(), "andy", add_word_boundary=False) == ["ANDY"]
    assert tokenize_sentencepiece_contexts(_FakeSentencePieceIds(), ["andy"])[0][0] == ord("▁")


def test_duplicate_contexts_are_deduplicated():
    graph = ContextGraph.from_token_ids([[1, 2], [1, 2], [2]])
    assert graph.context_count == 2
    assert graph.transition_count == 3


def test_large_single_token_dictionary_storage_is_linear():
    context_count = 10_000
    graph = ContextGraph.from_token_ids([[token] for token in range(1, context_count + 1)])

    assert graph.num_nodes == context_count + 1
    assert graph.transition_count == context_count
    assert all(not hasattr(state, "candidate_tokens") for state in graph.states)
    assert all(not hasattr(state, "matches") for state in graph.states)


def test_large_root_branching_is_not_materialized_as_injection_candidates():
    graph = ContextGraph.from_token_ids([[token] for token in range(1, 1_001)])
    active = graph.active_candidates(graph.root, maximum=12)

    assert active.tokens == ()
    assert active.competing_count == 13
    assert active.truncated


def test_active_phrase_continuation_remains_injectable_with_large_root():
    graph = ContextGraph.from_token_ids([[1, 50_000]] + [[token] for token in range(2, 1_002)])
    state = graph.forward_one_step(graph.root, 1)
    active = graph.active_candidates(state, maximum=12)

    assert active.tokens == (50_000,)
    assert active.truncated


def test_nested_completion_outputs_are_longest_first_and_bounded():
    policy = ContextPolicy(max_completed_contexts_per_token=3)
    graph = ContextGraph.from_token_ids(
        [[1] * length for length in range(1, 101)],
        policy=policy,
    )
    state = graph.root
    for _ in range(100):
        state = graph.forward_one_step(state, 1)

    assert tuple(graph.completed_matches(state)) == (
        (100, policy.completion_bonus),
        (99, policy.completion_bonus),
        (98, policy.completion_bonus),
    )


def test_persistent_context_history_uses_logarithmic_ancestor_lookup():
    states = [ContextScoreState()]
    for _ in range(4_096):
        states.append(states[-1].advance(()))

    ancestor, hops = states[-1]._ancestor(4_095)
    assert ancestor.length == 1
    assert hops <= (4_096).bit_length()
    assert max(len(state.jumps) for state in states) <= (4_096).bit_length()
    assert sum(len(state.jumps) for state in states) <= len(states) * (4_096).bit_length()
