import math
from itertools import product

import pytest
import torch

from asr_decoder import (
    AdaptiveGatingConfig,
    ContextPolicy,
    CTCDecoder,
    DecodingQuality,
    HotwordStrength,
)
from asr_decoder.prefix_score import append_alignment, materialize_alignment

SYMBOLS = {"<blank>": 0, "a": 1, "b": 2, "c": 3, "x": 4}


def log_probs(rows):
    return torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1)


def decode_in_chunks(decoder, probabilities, split, **kwargs):
    decoder.prefix_beam_search(probabilities[:split], **kwargs)
    return decoder.prefix_beam_search(probabilities[split:], finalize=True, **kwargs)


def test_streaming_is_invariant_for_every_chunk_boundary():
    probabilities = log_probs(
        [
            [1, 5, 0, 0, 0],
            [4, 1, 0, 0, 0],
            [0, 0, 5, 1, 0],
            [4, 0, 1, 0, 0],
            [0, 0, 0, 5, 1],
        ]
    )
    expected = CTCDecoder(["abc", "bc"], SYMBOLS).prefix_beam_search(
        probabilities,
        beam_size=4,
        token_beam_size=4,
        finalize=True,
        return_gating_diagnostics=True,
    )
    for split in range(1, len(probabilities)):
        actual = decode_in_chunks(
            CTCDecoder(["abc", "bc"], SYMBOLS),
            probabilities,
            split,
            beam_size=4,
            token_beam_size=4,
            return_gating_diagnostics=True,
        )
        assert actual == expected


def test_token_beam_is_independent_from_prefix_beam():
    result = CTCDecoder().prefix_beam_search(
        log_probs([[0, 5, 4, 3, 2], [5, 0, 0, 0, 0], [0, 2, 4, 5, 1]]),
        beam_size=2,
        token_beam_size=4,
        finalize=True,
    )
    assert len(result["tokens"]) == 2


def test_blank_is_considered_even_when_outside_token_topk():
    result = CTCDecoder().prefix_beam_search(
        log_probs([[0, 5, 4, 3, 2], [0, 5, 4, 3, 2]]),
        beam_size=3,
        token_beam_size=2,
        finalize=True,
    )
    assert [1] in result["tokens"]


def test_returned_probabilities_follow_selected_viterbi_path():
    result = CTCDecoder().prefix_beam_search(
        log_probs([[0, 5, 1, 0, 0], [4, 3, 0, 0, 0]]),
        beam_size=3,
        token_beam_size=3,
        return_token_probabilities=True,
        finalize=True,
    )
    assert len(result["tokens"][0]) == len(result["probs"][0])


@pytest.mark.parametrize(
    ("rows", "expected_span", "expected_log_probability"),
    [
        ([[-5.0, -2.0], [-0.1, -1.0]], (0, 1), -2.0),
        ([[-5.0, -2.0], [-1.0, -0.1]], (0, 2), -0.1),
    ],
)
def test_probabilities_and_timestamps_choose_the_same_viterbi_ending_path(
    rows,
    expected_span,
    expected_log_probability,
):
    result = CTCDecoder(frame_shift_ms=10).prefix_beam_search(
        torch.tensor(rows),
        beam_size=4,
        token_beam_size=2,
        return_token_probabilities=True,
        finalize=True,
    )
    hypothesis = result["tokens"].index([1])
    timestamp = result["timestamps"][hypothesis][0]
    assert (timestamp["start_frame"], timestamp["end_frame"]) == expected_span
    assert result["probs"][hypothesis] == pytest.approx([math.exp(expected_log_probability)])


def _reference_ctc_alignment(path, rows, blank_id=0):
    tokens, spans, token_log_probabilities = [], [], []
    previous = None
    score = 0.0
    for frame, token_id in enumerate(path):
        probability = rows[frame][token_id]
        score += probability
        if token_id == blank_id:
            previous = None
        elif token_id == previous:
            spans[-1] = (spans[-1][0], frame + 1)
            token_log_probabilities[-1] = max(token_log_probabilities[-1], probability)
        else:
            tokens.append(token_id)
            spans.append((frame, frame + 1))
            token_log_probabilities.append(probability)
            previous = token_id
    return tuple(tokens), score, spans, token_log_probabilities


def test_prefix_alignment_matches_exhaustive_ctc_paths():
    vocabulary_size = 3
    for frame_count in range(1, 5):
        rows = [
            [-((token_id + 1) * (10 ** (-frame))) for token_id in range(vocabulary_size)]
            for frame in range(frame_count)
        ]
        expected = {}
        for path in product(range(vocabulary_size), repeat=frame_count):
            tokens, score, spans, token_log_probabilities = _reference_ctc_alignment(path, rows)
            if tokens not in expected or score > expected[tokens][0]:
                expected[tokens] = (score, spans, token_log_probabilities)

        result = CTCDecoder(frame_shift_ms=2.5).prefix_beam_search(
            torch.tensor(rows, dtype=torch.float32),
            beam_size=1_000,
            token_beam_size=vocabulary_size,
            return_token_probabilities=True,
            finalize=True,
        )
        actual = {tuple(tokens): index for index, tokens in enumerate(result["tokens"])}
        assert actual.keys() == expected.keys()
        for tokens, (_, spans, token_log_probabilities) in expected.items():
            index = actual[tokens]
            timestamps = result["timestamps"][index]
            assert [(item["start_frame"], item["end_frame"]) for item in timestamps] == spans
            assert [item["start_ms"] for item in timestamps] == [start * 2.5 for start, _ in spans]
            assert [item["end_ms"] for item in timestamps] == [end * 2.5 for _, end in spans]
            assert result["probs"][index] == pytest.approx(
                [math.exp(probability) for probability in token_log_probabilities]
            )


def test_impossible_paths_are_not_returned_as_nbest_hypotheses():
    result = CTCDecoder().prefix_beam_search(
        torch.tensor([[0.0, float("-inf"), float("-inf")]]),
        beam_size=3,
        token_beam_size=3,
        finalize=True,
    )
    assert result["tokens"] == [[]]


def test_create_stream_returns_independent_state():
    template = CTCDecoder(["ab"], SYMBOLS)
    first, second = template.create_stream(), template.create_stream()
    first.prefix_beam_search(log_probs([[0, 5, 1, 0, 0]]), beam_size=2)
    assert first.processed_frames == 1
    assert second.processed_frames == 0


def test_empty_contexts_keep_contextual_decoder_disabled():
    assert CTCDecoder([], SYMBOLS).context_graph is None
    assert CTCDecoder(context_token_ids=[[]]).context_graph is None


@pytest.mark.parametrize(
    ("strength", "factory"),
    [
        (HotwordStrength.CONSERVATIVE, ContextPolicy.conservative),
        (HotwordStrength.BALANCED, ContextPolicy.balanced),
        (HotwordStrength.AGGRESSIVE, ContextPolicy.aggressive),
        (HotwordStrength.LOW_FALSE_ACTIVATION, ContextPolicy.low_false_activation),
    ],
)
def test_hotword_strength_selects_a_complete_named_policy(strength, factory):
    decoder = CTCDecoder(["ab"], SYMBOLS, hotword_strength=strength)
    assert decoder.context_policy == factory()
    assert decoder.context_graph.policy is decoder.context_policy


def test_advanced_policy_and_user_strength_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        CTCDecoder(
            context_token_ids=[[1, 2]],
            context_policy=ContextPolicy(),
            hotword_strength=HotwordStrength.BALANCED,
        )


def test_context_token_ids_use_model_tokenization_directly():
    assert CTCDecoder(context_token_ids=[[1, 2]]).context_graph.num_nodes == 3


def test_text_and_token_id_contexts_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        CTCDecoder(["ab"], SYMBOLS, context_token_ids=[[1, 2]])

    with pytest.raises(ValueError, match="not both"):
        CTCDecoder([], SYMBOLS, context_token_ids=[])


@pytest.mark.parametrize("token_ids", [[[1, -1]], [[1, 2.5]]])
def test_context_token_ids_are_validated(token_ids):
    with pytest.raises((TypeError, ValueError)):
        CTCDecoder(context_token_ids=token_ids)


def test_bool_is_not_accepted_as_a_token_id():
    with pytest.raises(TypeError, match="integers"):
        CTCDecoder(context_token_ids=[[True]])
    with pytest.raises(TypeError, match="blank_id must be an integer"):
        CTCDecoder(blank_id=True)


def test_context_token_ids_must_not_include_blank():
    with pytest.raises(ValueError, match="blank_id"):
        CTCDecoder(context_token_ids=[[0, 1]], blank_id=0)


def test_word_boundary_tokens_suppress_non_boundary_context_continuations():
    # The ordinary path is ``and ▁they``.  A context ``and y`` must not
    # inject ``y`` on the frame where the acoustic winner is a word start.
    probabilities = log_probs(
        [
            [0, 8, 0, 0, 0],  # and
            [0, 0, 6, 8, 7],  # ▁they wins; y is only a context injection
        ]
    )
    result = CTCDecoder(
        context_token_ids=[[1, 2]],
        word_boundary_token_ids={3},
        context_policy=ContextPolicy(
            completion_bonus=1.0,
            context_token_prune_threshold=3.0,
            gating=AdaptiveGatingConfig(enabled=False),
        ),
    ).prefix_beam_search(
        probabilities,
        beam_size=4,
        token_beam_size=2,
        finalize=True,
        return_gating_diagnostics=True,
    )
    assert result["tokens"][0] == [1, 3]
    assert result["gating"]["boundary_suppressed_context_candidates"] > 0


@pytest.mark.parametrize("value", [[True], [1.5], [-1]])
def test_word_boundary_token_ids_are_validated(value):
    with pytest.raises((TypeError, ValueError)):
        CTCDecoder(word_boundary_token_ids=value)


def test_create_stream_inherits_word_boundary_token_ids():
    decoder = CTCDecoder(word_boundary_token_ids={3})
    assert decoder.create_stream().word_boundary_token_ids == frozenset({3})


def test_greedy_search_uses_argmax_and_standard_ctc_collapse():
    probabilities = log_probs([[0, 5, 1, 0, 0], [0, 5, 1, 0, 0], [5, 0, 1, 0, 0], [0, 5, 1, 0, 0], [0, 1, 5, 0, 0]])
    result = CTCDecoder().greedy_search(probabilities, finalize=True)
    assert result["tokens"] == [1, 1, 2]
    assert [(item["start_frame"], item["end_frame"]) for item in result["timestamps"]] == [
        (0, 2),
        (3, 4),
        (4, 5),
    ]


@pytest.mark.parametrize("search_name", ["greedy_search", "prefix_beam_search"])
def test_timestamps_cover_repeats_blank_separated_tokens_and_milliseconds(search_name):
    probabilities = log_probs(
        [
            [-5, 10, -5],
            [-5, 10, -5],
            [10, -5, -5],
            [-5, 10, -5],
            [-5, -5, 10],
            [-5, -5, 10],
        ]
    )
    result = getattr(CTCDecoder(frame_shift_ms=20), search_name)(probabilities, finalize=True)
    if search_name == "prefix_beam_search":
        assert result["tokens"][0] == [1, 1, 2]
        timestamps = result["timestamps"][0]
    else:
        assert result["tokens"] == [1, 1, 2]
        timestamps = result["timestamps"]

    assert timestamps == [
        {"start_frame": 0, "end_frame": 2, "start_ms": 0.0, "end_ms": 40.0},
        {"start_frame": 3, "end_frame": 4, "start_ms": 60.0, "end_ms": 80.0},
        {"start_frame": 4, "end_frame": 6, "start_ms": 80.0, "end_ms": 120.0},
    ]


@pytest.mark.parametrize("search_name", ["greedy_search", "prefix_beam_search"])
def test_timestamps_without_frame_shift_keep_frame_spans(search_name):
    result = getattr(CTCDecoder(), search_name)(log_probs([[0, 5], [0, 5]]), finalize=True)
    timestamps = result["timestamps"][0] if search_name == "prefix_beam_search" else result["timestamps"]
    assert timestamps[0] == {"start_frame": 0, "end_frame": 2, "start_ms": None, "end_ms": None}


@pytest.mark.parametrize("search_name", ["greedy_search", "prefix_beam_search"])
def test_empty_input_has_empty_timestamps(search_name):
    result = getattr(CTCDecoder(frame_shift_ms=10), search_name)(torch.empty((0, 3)), finalize=True)
    assert result["timestamps"] == ([[]] if search_name == "prefix_beam_search" else [])


@pytest.mark.parametrize(
    "value",
    [True, False, "10", object()],
)
def test_frame_shift_rejects_non_real_values(value):
    with pytest.raises(TypeError, match="real number"):
        CTCDecoder(frame_shift_ms=value)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_frame_shift_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        CTCDecoder(frame_shift_ms=value)


@pytest.mark.parametrize("search_name", ["greedy_search", "prefix_beam_search"])
def test_timestamp_streaming_is_invariant_at_every_chunk_boundary(search_name):
    probabilities = log_probs([[0, 6, 1], [0, 6, 1], [6, 0, 1], [0, 6, 1], [0, 1, 6], [0, 1, 6]])
    expected = getattr(CTCDecoder(frame_shift_ms=7.5), search_name)(probabilities, finalize=True)
    for split in range(1, len(probabilities)):
        stream = CTCDecoder(frame_shift_ms=7.5).create_stream()
        getattr(stream, search_name)(probabilities[:split])
        actual = getattr(stream, search_name)(probabilities[split:], finalize=True)
        assert actual == expected


def test_create_stream_inherits_frame_shift_and_finalize_restarts_global_frames():
    stream = CTCDecoder(frame_shift_ms=12.5).create_stream()
    assert stream.frame_shift_ms == 12.5
    first = stream.greedy_search(log_probs([[0, 5], [0, 5]]), finalize=True)
    second = stream.greedy_search(log_probs([[0, 5]]), finalize=True)
    assert first["timestamps"][0]["end_ms"] == 25.0
    assert second["timestamps"][0]["start_frame"] == 0
    assert second["timestamps"][0]["end_ms"] == 12.5


def test_greedy_streaming_matches_one_shot():
    probabilities = log_probs([[0, 5, 1], [5, 0, 0], [0, 1, 5]])
    expected = CTCDecoder().greedy_search(probabilities, finalize=True)
    stream = CTCDecoder().create_stream()
    stream.greedy_search(probabilities[:2])
    assert stream.greedy_search(probabilities[2:], finalize=True) == expected


def test_scores_name_contextual_bias_and_add_up():
    result = CTCDecoder(["ab"], SYMBOLS).prefix_beam_search(
        log_probs([[0, 5, 1, 0, 0], [5, 0, 0, 0, 0], [0, 1, 5, 0, 0]]),
        beam_size=3,
        finalize=True,
        return_scores=True,
    )
    score = result["scores"][0]
    assert set(score) == {"acoustic", "contextual_bias", "total"}
    assert score["total"] == score["acoustic"] + score["contextual_bias"]


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        (DecodingQuality.LOW_LATENCY, (4, 6)),
        (DecodingQuality.BALANCED, (8, 6)),
        (DecodingQuality.HIGH_ACCURACY, (12, 6)),
    ],
)
def test_decoding_quality_selects_bounded_beam_defaults(quality, expected):
    result = CTCDecoder(decoding_quality=quality).prefix_beam_search(
        log_probs([[0, 5, 4, 3, 2, 1]]),
        finalize=True,
        return_gating_diagnostics=True,
    )
    gating = result["gating"]
    assert (gating["beam_size"], gating["token_beam_size"]) == expected


def test_search_configuration_is_locked_for_a_stream_and_omission_reuses_it():
    probabilities = log_probs([[0, 5, 4], [5, 0, 0]])
    decoder = CTCDecoder()
    decoder.prefix_beam_search(probabilities[:1], beam_size=2, token_beam_size=2)
    decoder.prefix_beam_search(probabilities[1:], finalize=True)

    decoder.prefix_beam_search(probabilities[:1], beam_size=2, token_beam_size=2)
    with pytest.raises(ValueError, match="beam_size cannot change"):
        decoder.prefix_beam_search(probabilities[1:], beam_size=3)


def test_token_beam_configuration_cannot_change_within_stream():
    decoder = CTCDecoder()
    decoder.prefix_beam_search(log_probs([[0, 5, 4]]), token_beam_size=2)
    with pytest.raises(ValueError, match="token_beam_size cannot change"):
        decoder.prefix_beam_search(log_probs([[5, 0, 0]]), token_beam_size=3)


def test_vocabulary_size_cannot_change_within_stream():
    decoder = CTCDecoder()
    decoder.prefix_beam_search(log_probs([[0, 5, 4]]))
    with pytest.raises(ValueError, match="vocabulary_size cannot change"):
        decoder.prefix_beam_search(log_probs([[5, 0, 0, 0]]))


def test_no_context_skips_all_adaptive_gating_and_candidate_work():
    result = CTCDecoder().prefix_beam_search(
        log_probs([[0, 5, 4], [5, 0, 0]]),
        finalize=True,
        return_gating_diagnostics=True,
    )
    diagnostics = result["gating"]
    assert diagnostics["contextual_biasing"] is False
    assert diagnostics["adaptive_context_gating"] is False
    assert diagnostics["gating_frames"] == 0
    assert diagnostics["mean_gating_acoustic_scale"] is None
    assert diagnostics["evaluated_context_candidates"] == 0


def test_disabled_gating_has_unambiguous_diagnostics():
    policy = ContextPolicy(gating=AdaptiveGatingConfig(enabled=False))
    result = CTCDecoder(context_token_ids=[[1]], context_policy=policy).prefix_beam_search(
        log_probs([[0, 5, 4]]),
        finalize=True,
        return_gating_diagnostics=True,
    )
    diagnostics = result["gating"]
    assert diagnostics["contextual_biasing"] is True
    assert diagnostics["adaptive_context_gating"] is False
    assert diagnostics["gating_frames"] == 0
    assert diagnostics["mean_gating_acoustic_scale"] is None


def test_adaptive_gating_tracks_logit_scale_without_changing_result():
    rows = torch.tensor([[0, 5, 4.2, 0, 0, 3.8], [5, 0, 0, 0, 0, 0], [0, 0, 5, 4.2, 0, 3.8]])
    outputs, scales = [], []
    for scale in (0.5, 1.0, 2.0):
        result = CTCDecoder(context_token_ids=[[5, 5]]).prefix_beam_search(
            torch.log_softmax(rows * scale, dim=-1),
            finalize=True,
            return_gating_diagnostics=True,
        )
        outputs.append(result["tokens"][0])
        scales.append(result["gating"]["mean_gating_acoustic_scale"])
    assert outputs[0] == outputs[1] == outputs[2]
    assert scales[0] < scales[1] < scales[2]


def test_context_reward_is_prefix_deterministic_across_acoustic_scales_and_alignments():
    policy = ContextPolicy(completion_bonus=5.0)
    inputs = [
        log_probs([[0, 8, 0], [8, 0, 0]]),
        log_probs([[0, 1.5, 1.4], [0, 1.5, 1.4], [8, 0, 0]]),
    ]
    biases = []
    for probabilities in inputs:
        result = CTCDecoder(context_token_ids=[[1]], context_policy=policy).prefix_beam_search(
            probabilities,
            finalize=True,
            return_scores=True,
        )
        index = result["tokens"].index([1])
        biases.append(result["scores"][index]["contextual_bias"])
    assert biases == [5.0, 5.0]


def test_unreachable_context_does_not_change_base_transcription():
    probabilities = torch.log_softmax(
        torch.tensor([[0.0, 8.0, 2.0, 1.0, float("-inf")], [8.0, 0.0, 0.0, 0.0, float("-inf")]]),
        dim=-1,
    )
    baseline = CTCDecoder().prefix_beam_search(probabilities, finalize=True)
    contextual = CTCDecoder(context_token_ids=[[4]]).prefix_beam_search(probabilities, finalize=True)
    assert contextual == baseline


def test_zero_completion_policy_does_not_change_base_search_without_injection():
    probabilities = log_probs([[0, 8, 2], [8, 0, 0], [0, 2, 8]])
    baseline = CTCDecoder().prefix_beam_search(probabilities, finalize=True)
    policy = ContextPolicy(completion_bonus=0.0, max_injected_candidates=0)
    contextual = CTCDecoder(context_token_ids=[[1, 2]], context_policy=policy).prefix_beam_search(
        probabilities,
        finalize=True,
    )
    assert contextual == baseline


def test_zero_completion_policy_disables_default_injection_regression():
    generator = torch.Generator().manual_seed(2)
    probabilities = torch.log_softmax(torch.randn(5, 5, generator=generator), dim=-1)
    baseline = CTCDecoder().prefix_beam_search(
        probabilities,
        beam_size=3,
        token_beam_size=2,
        finalize=True,
    )
    decoder = CTCDecoder(
        context_token_ids=[[4]],
        context_policy=ContextPolicy(completion_bonus=0.0),
    )
    contextual = decoder.prefix_beam_search(
        probabilities,
        beam_size=3,
        token_beam_size=2,
        finalize=True,
        return_gating_diagnostics=True,
    )

    assert baseline["tokens"][0] == [1, 1, 4]
    assert contextual["tokens"] == baseline["tokens"]
    assert contextual["timestamps"] == baseline["timestamps"]
    assert decoder.context_graph is None
    assert contextual["gating"]["contextual_biasing"] is False
    assert contextual["gating"]["adaptive_context_gating"] is False
    assert contextual["gating"]["evaluated_context_candidates"] == 0


@pytest.mark.parametrize("seed", range(20))
def test_zero_completion_policy_is_a_randomized_search_no_op(seed):
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.log_softmax(torch.randn(5, 5, generator=generator), dim=-1)
    baseline = CTCDecoder().prefix_beam_search(
        probabilities,
        beam_size=3,
        token_beam_size=2,
        finalize=True,
    )
    contextual = CTCDecoder(
        context_token_ids=[[(seed % 4) + 1]],
        context_policy=ContextPolicy(completion_bonus=0.0),
    ).prefix_beam_search(
        probabilities,
        beam_size=3,
        token_beam_size=2,
        finalize=True,
    )
    assert contextual == baseline


@pytest.mark.parametrize("token_ids", [[[True]], [[-1]], [[1.5]], [[0]]])
def test_zero_completion_policy_still_validates_context_token_ids(token_ids):
    policy = ContextPolicy(completion_bonus=0.0)
    with pytest.raises((TypeError, ValueError)):
        CTCDecoder(context_token_ids=token_ids, context_policy=policy)


def test_zero_completion_policy_still_requires_text_tokenizer_contract():
    with pytest.raises(ValueError, match="symbol_table is required"):
        CTCDecoder(contexts=["a"], context_policy=ContextPolicy(completion_bonus=0.0))


def test_zero_completion_policy_still_validates_context_vocabulary():
    decoder = CTCDecoder(
        context_token_ids=[[10]],
        context_policy=ContextPolicy(completion_bonus=0.0),
    )
    with pytest.raises(ValueError, match="context token IDs"):
        decoder.prefix_beam_search(log_probs([[0, 1, 2]]))


def test_strong_blank_evidence_prevents_hotword_mistrigger():
    result = CTCDecoder(
        context_token_ids=[[5]],
        hotword_strength=HotwordStrength.AGGRESSIVE,
    ).prefix_beam_search(log_probs([[8, 0, 0, 0, 0, 0]] * 2), finalize=True)
    assert result["tokens"][0] == []


def test_context_candidate_evaluation_is_strictly_bounded_for_large_dictionary():
    policy = ContextPolicy(max_injected_candidates=8)
    decoder = CTCDecoder(
        context_token_ids=[[token] for token in range(1, 1_001)],
        context_policy=policy,
    )
    result = decoder.prefix_beam_search(
        log_probs([[0] * 1_001]),
        beam_size=2,
        token_beam_size=2,
        finalize=True,
        return_gating_diagnostics=True,
    )
    diagnostics = result["gating"]
    assert diagnostics["max_context_candidates_per_frame"] <= 8
    assert diagnostics["evaluated_context_candidates"] <= 8
    assert diagnostics["gathered_context_candidates"] <= 8
    assert diagnostics["context_candidate_limit_hits"] == 1


def test_gating_competition_is_independent_of_duplicate_hypotheses_at_same_state():
    contexts = [[token] for token in range(5, 10)]
    probabilities = log_probs([[0, 8, 7, 6, 5, 0, 0, 0, 0, 0]] * 2)
    diagnostics = []
    for beam_size in (2, 8):
        result = CTCDecoder(context_token_ids=contexts).prefix_beam_search(
            probabilities,
            beam_size=beam_size,
            token_beam_size=4,
            finalize=True,
            return_gating_diagnostics=True,
        )
        diagnostics.append(result["gating"])
    assert diagnostics[0]["mean_gating_factor"] == diagnostics[1]["mean_gating_factor"]
    assert diagnostics[0]["max_context_candidates_per_frame"] == diagnostics[1]["max_context_candidates_per_frame"]


def test_active_continuation_can_be_injected_without_scanning_whole_dictionary():
    contexts = [[1, 999]] + [[token] for token in range(2, 102)]
    policy = ContextPolicy(completion_bonus=6.0, max_injected_candidates=4)
    result = CTCDecoder(context_token_ids=contexts, context_policy=policy).prefix_beam_search(
        log_probs([[0, 7, 6] + [0] * 997, [0, 7, 6] + [0] * 996 + [5.8]]),
        beam_size=4,
        token_beam_size=2,
        finalize=True,
        return_gating_diagnostics=True,
    )
    assert any(999 in tokens for tokens in result["tokens"])
    assert result["gating"]["max_context_candidates_per_frame"] <= 4


def test_overlapping_contexts_are_not_double_counted():
    policy = ContextPolicy(
        completion_bonus=5.0,
        gating=AdaptiveGatingConfig(enabled=False),
    )
    result = CTCDecoder(context_token_ids=[[1], [1, 2]], context_policy=policy).prefix_beam_search(
        log_probs([[0, 8, 0], [8, 0, 0], [0, 0, 8]]),
        finalize=True,
        return_scores=True,
    )
    index = result["tokens"].index([1, 2])
    assert result["scores"][index]["contextual_bias"] == pytest.approx(5.0)


def test_future_blank_frames_do_not_rescale_completed_context_history():
    policy = ContextPolicy(gating=AdaptiveGatingConfig(enabled=True))
    decoder = CTCDecoder(context_token_ids=[[1]], context_policy=policy)
    partial = decoder.prefix_beam_search(
        log_probs([[0, 8, 0]]),
        return_scores=True,
    )
    partial_index = partial["tokens"].index([1])
    original_bias = partial["scores"][partial_index]["contextual_bias"]
    final = decoder.prefix_beam_search(
        log_probs([[12, 0, 0]] * 50),
        finalize=True,
        return_scores=True,
    )
    final_index = final["tokens"].index([1])
    assert final["scores"][final_index]["contextual_bias"] == original_bias


def test_long_streaming_partial_results_do_not_run_quadratic_final_rescoring():
    decoder = CTCDecoder(context_token_ids=[[1, 2]])
    chunk = log_probs([[0, 8, 0], [8, 0, 0], [0, 0, 8], [8, 0, 0]])
    for _ in range(100):
        result = decoder.prefix_beam_search(chunk)
        assert result["tokens"]
    assert not hasattr(decoder.context_graph, "final_score")


def test_context_vocabulary_validation_does_not_depend_on_gating():
    policy = ContextPolicy(gating=AdaptiveGatingConfig(enabled=False))
    decoder = CTCDecoder(context_token_ids=[[10]], context_policy=policy)
    with pytest.raises(ValueError, match="context token IDs"):
        decoder.prefix_beam_search(log_probs([[0, 1, 2]]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_policy_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite"):
        ContextPolicy(completion_bonus=value)
    with pytest.raises(ValueError, match="finite"):
        AdaptiveGatingConfig(reference_top_spread=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_decoder_rejects_nan_and_positive_infinity(value):
    with pytest.raises(ValueError, match="finite log probability"):
        CTCDecoder().prefix_beam_search(torch.tensor([[0.0, value]]))


def test_decoder_rejects_non_floating_input_for_both_searches():
    probabilities = torch.tensor([[0, 1]])
    for search in (CTCDecoder().greedy_search, CTCDecoder().prefix_beam_search):
        with pytest.raises(TypeError, match="floating-point"):
            search(probabilities)


def test_all_negative_infinity_frames_are_rejected_consistently():
    probabilities = torch.full((1, 3), float("-inf"))
    for search in (CTCDecoder().greedy_search, CTCDecoder().prefix_beam_search):
        with pytest.raises(ValueError, match="finite log probability"):
            search(probabilities)


def test_partial_output_is_detached_from_stream_history():
    probabilities = log_probs([[0, 8, 0], [8, 0, 0], [0, 0, 8]])
    first = CTCDecoder().create_stream()
    control = CTCDecoder().create_stream()
    partial = first.prefix_beam_search(
        probabilities[:1],
        return_token_probabilities=True,
    )
    control.prefix_beam_search(probabilities[:1], return_token_probabilities=True)
    partial["timestamps"][0][0]["end_frame"] = 999
    partial["timestamps"][0].append({"start_frame": 999})
    partial["probs"][0].append(0.0)

    actual = first.prefix_beam_search(
        probabilities[1:],
        finalize=True,
        return_token_probabilities=True,
    )
    expected = control.prefix_beam_search(
        probabilities[1:],
        finalize=True,
        return_token_probabilities=True,
    )
    assert actual == expected


def test_greedy_partial_timestamps_are_detached_from_stream_history():
    probabilities = log_probs([[0, 8, 0], [0, 8, 0], [8, 0, 0]])
    first = CTCDecoder(frame_shift_ms=10)
    control = CTCDecoder(frame_shift_ms=10)
    partial = first.greedy_search(probabilities[:1])
    control.greedy_search(probabilities[:1])
    partial["timestamps"][0]["end_frame"] = 999
    partial["timestamps"].append({"start_frame": 999})

    actual = first.greedy_search(probabilities[1:], finalize=True)
    expected = control.greedy_search(probabilities[1:], finalize=True)
    assert actual == expected


def test_blank_paths_share_immutable_alignment_history():
    decoder = CTCDecoder()
    decoder.prefix_beam_search(log_probs([[0, 8, 0]]), beam_size=3)
    score = dict(decoder.hypotheses)[(1,)]
    alignment = score.non_blank_alignment
    for _ in range(100):
        decoder.prefix_beam_search(log_probs([[8, 0, 0]]))
        score = dict(decoder.hypotheses)[(1,)]
        assert score.alignment() is alignment


def test_persistent_alignment_append_and_extend_only_replace_the_tail_node():
    first = append_alignment(None, token_id=1, frame=0, log_probability=-2.0)
    second = append_alignment(first, token_id=2, frame=1, log_probability=-3.0)
    extended = second.extend(end_frame=3, log_probability=-1.0)

    assert second.parent is first
    assert extended.parent is first
    assert extended.length == second.length == 2
    assert materialize_alignment(extended) == (
        [1, 2],
        [(0, 1), (1, 3)],
        [-2.0, -1.0],
    )


def test_decoder_mode_is_locked_greedy_then_prefix_until_reset():
    decoder = CTCDecoder()
    decoder.greedy_search(log_probs([[0, 1]]))
    with pytest.raises(ValueError, match="cannot switch from greedy to prefix_beam"):
        decoder.prefix_beam_search(log_probs([[0, 1]]))
    decoder.reset()
    decoder.prefix_beam_search(log_probs([[0, 1]]), finalize=True)


def test_decoder_mode_is_locked_prefix_then_greedy_until_finalize():
    decoder = CTCDecoder()
    decoder.prefix_beam_search(log_probs([[0, 1]]))
    with pytest.raises(ValueError, match="cannot switch from prefix_beam to greedy"):
        decoder.greedy_search(log_probs([[0, 1]]))
    decoder.prefix_beam_search(log_probs([[0, 1]]), finalize=True)
    decoder.greedy_search(log_probs([[0, 1]]), finalize=True)


def test_invalid_prefix_call_does_not_lock_or_mutate_stream():
    decoder = CTCDecoder()
    initial_hypothesis = decoder.hypotheses[0]
    with pytest.raises(ValueError, match="finite log probability"):
        decoder.prefix_beam_search(
            torch.full((1, 3), float("-inf")),
            beam_size=2,
            token_beam_size=2,
        )

    assert decoder._decode_mode is None
    assert decoder._search_config is None
    assert decoder.processed_frames == 0
    assert decoder.hypotheses[0] is initial_hypothesis
    result = decoder.greedy_search(log_probs([[0, 1, 0]]), finalize=True)
    assert result["tokens"] == [1]


def test_invalid_greedy_call_does_not_lock_or_mutate_stream():
    decoder = CTCDecoder()
    initial_hypothesis = decoder.hypotheses[0]
    with pytest.raises(ValueError, match="finite log probability"):
        decoder.greedy_search(torch.full((1, 3), float("-inf")))

    assert decoder._decode_mode is None
    assert decoder._search_config is None
    assert decoder.processed_frames == 0
    assert decoder.hypotheses[0] is initial_hypothesis
    result = decoder.prefix_beam_search(log_probs([[0, 1, 0]]), finalize=True)
    assert result["tokens"][0] == [1]


@pytest.mark.parametrize("value", [1.5, float("nan"), float("inf"), float("-inf")])
def test_integer_decoder_parameters_reject_all_floats(value):
    with pytest.raises(TypeError, match="blank_id must be an integer"):
        CTCDecoder(blank_id=value)
    with pytest.raises(TypeError, match="beam sizes must be integers"):
        CTCDecoder().prefix_beam_search(log_probs([[0, 1]]), beam_size=value)


@pytest.mark.parametrize(
    ("decoder", "probabilities", "message"),
    [
        (CTCDecoder(blank_id=5), torch.empty((2, 5)), "blank_id"),
        (CTCDecoder(), torch.empty((2, 0)), "vocabulary_size"),
    ],
)
def test_invalid_decoder_input_is_rejected(decoder, probabilities, message):
    with pytest.raises(ValueError, match=message):
        decoder.greedy_search(probabilities)
