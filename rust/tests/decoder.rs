use std::collections::{HashMap, HashSet};

use asr_decoder::{
    AdaptiveGatingConfig, CTCDecoder, ContextPolicy, DecodeResult, DecoderConfig, DecoderError,
    LogProbabilities, PrefixBeamSearchOptions,
};

type ReferenceAlignment = (Vec<usize>, f64, Vec<(usize, usize)>, Vec<f64>);
type ExpectedAlignments = HashMap<Vec<usize>, (f64, Vec<(usize, usize)>, Vec<f64>)>;

fn beam_options() -> PrefixBeamSearchOptions {
    PrefixBeamSearchOptions {
        beam_size: Some(1000),
        token_beam_size: Some(3),
        finalize: true,
        return_token_probabilities: true,
        ..PrefixBeamSearchOptions::default()
    }
}

#[test]
fn alignments_match_exhaustive_ctc_paths() {
    for frame_count in 1..=4 {
        let rows: Vec<Vec<f32>> = (0..frame_count)
            .map(|frame| {
                (0..3)
                    .map(|token| -((token + 1) as f32) * 10_f32.powi(-(frame as i32)))
                    .collect()
            })
            .collect();
        let flat: Vec<_> = rows.iter().flatten().copied().collect();
        let mut expected = ExpectedAlignments::new();
        for encoded in 0..3_usize.pow(frame_count as u32) {
            let mut value = encoded;
            let mut path = vec![0; frame_count];
            for token in &mut path {
                *token = value % 3;
                value /= 3;
            }
            let alignment = reference_alignment(&path, &rows);
            let current =
                expected
                    .entry(alignment.0)
                    .or_insert((f64::NEG_INFINITY, Vec::new(), Vec::new()));
            if alignment.1 > current.0 {
                *current = (alignment.1, alignment.2, alignment.3);
            }
        }

        let input = LogProbabilities::new(&flat, frame_count, 3).unwrap();
        let mut decoder = CTCDecoder::new(DecoderConfig {
            frame_shift_ms: Some(2.5),
            ..DecoderConfig::default()
        })
        .unwrap();
        let result = decoder.prefix_beam_search(input, beam_options()).unwrap();
        let actual: HashMap<_, _> = result
            .tokens
            .iter()
            .enumerate()
            .map(|(position, tokens)| (tokens.clone(), position))
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (tokens, (_, spans, log_probabilities)) in expected {
            let position = actual[&tokens];
            let actual_spans: Vec<_> = result.timestamps[position]
                .iter()
                .map(|timestamp| (timestamp.start_frame, timestamp.end_frame))
                .collect();
            assert_eq!(actual_spans, spans);
            if let Some(first) = spans.first() {
                assert_eq!(
                    result.timestamps[position][0].start_ms,
                    Some(first.0 as f64 * 2.5)
                );
            }
            for (actual, expected) in result.probabilities.as_ref().unwrap()[position]
                .iter()
                .zip(log_probabilities)
            {
                assert!((actual - expected.exp()).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn overlapping_contexts_are_not_double_counted() {
    let values = [-8.0, -0.1, -8.0, -0.1, -8.0, -8.0, -8.0, -8.0, -0.1];
    let input = LogProbabilities::new(&values, 3, 3).unwrap();
    let mut decoder = CTCDecoder::new(DecoderConfig {
        context_token_ids: vec![vec![1], vec![1, 2]],
        context_policy: Some(ContextPolicy {
            completion_bonus: 5.0,
            gating: AdaptiveGatingConfig {
                enabled: false,
                ..AdaptiveGatingConfig::default()
            },
            ..ContextPolicy::conservative()
        }),
        ..DecoderConfig::default()
    })
    .unwrap();
    let result = decoder
        .prefix_beam_search(
            input,
            PrefixBeamSearchOptions {
                return_scores: true,
                finalize: true,
                ..PrefixBeamSearchOptions::default()
            },
        )
        .unwrap();
    let position = result
        .tokens
        .iter()
        .position(|tokens| tokens == &[1, 2])
        .unwrap();
    assert_eq!(result.scores.unwrap()[position].contextual_bias, 5.0);
}

#[test]
fn word_boundary_suppresses_in_word_context_continuation() {
    let values = [-8.0, -0.1, -8.0, -8.0, -8.0, -8.0, -8.0, -2.1, -0.1, -1.1];
    let input = LogProbabilities::new(&values, 2, 5).unwrap();
    let mut decoder = CTCDecoder::new(DecoderConfig {
        context_token_ids: vec![vec![1, 2]],
        context_policy: Some(ContextPolicy {
            completion_bonus: 1.0,
            context_token_prune_threshold: 3.0,
            gating: AdaptiveGatingConfig {
                enabled: false,
                ..AdaptiveGatingConfig::default()
            },
            ..ContextPolicy::conservative()
        }),
        word_boundary_token_ids: HashSet::from([3]),
        ..DecoderConfig::default()
    })
    .unwrap();
    let result = decoder
        .prefix_beam_search(
            input,
            PrefixBeamSearchOptions {
                token_beam_size: Some(2),
                return_gating_diagnostics: true,
                ..PrefixBeamSearchOptions::default()
            },
        )
        .unwrap();
    assert_eq!(result.tokens[0], vec![1, 3]);
    assert!(
        result
            .gating
            .unwrap()
            .boundary_suppressed_context_candidates
            > 0
    );
}

#[test]
fn invalid_input_does_not_lock_stream() {
    let invalid_values = [f32::NEG_INFINITY; 3];
    let invalid = LogProbabilities::new(&invalid_values, 1, 3).unwrap();
    let mut decoder = CTCDecoder::new(DecoderConfig::default()).unwrap();
    assert!(matches!(
        decoder.prefix_beam_search(invalid, PrefixBeamSearchOptions::default()),
        Err(DecoderError::InvalidInput(_))
    ));

    let valid_values = [-2.0, -0.1, -3.0];
    let valid = LogProbabilities::new(&valid_values, 1, 3).unwrap();
    let result = decoder.greedy_search(valid, true, false).unwrap();
    assert_eq!(result.tokens, vec![vec![1]]);
}

#[test]
fn zero_completion_bonus_disables_bias_but_keeps_vocabulary_validation() {
    let mut decoder = CTCDecoder::new(DecoderConfig {
        context_token_ids: vec![vec![10]],
        context_policy: Some(ContextPolicy {
            completion_bonus: 0.0,
            ..ContextPolicy::conservative()
        }),
        ..DecoderConfig::default()
    })
    .unwrap();
    let values = [-2.0, -0.1, -3.0];
    let input = LogProbabilities::new(&values, 1, 3).unwrap();
    assert!(matches!(
        decoder.prefix_beam_search(input, PrefixBeamSearchOptions::default()),
        Err(DecoderError::InvalidInput(_))
    ));
}

#[test]
fn empty_input_returns_the_current_empty_hypothesis() {
    let input = LogProbabilities::new(&[], 0, 3).unwrap();
    let mut decoder = CTCDecoder::new(DecoderConfig::default()).unwrap();
    let result: DecodeResult = decoder
        .prefix_beam_search(input, PrefixBeamSearchOptions::default())
        .unwrap();
    assert_eq!(result.tokens, vec![Vec::<usize>::new()]);
    assert_eq!(result.timestamps, vec![vec![]]);
}

fn reference_alignment(path: &[usize], rows: &[Vec<f32>]) -> ReferenceAlignment {
    let mut tokens = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut log_probabilities: Vec<f64> = Vec::new();
    let mut previous = None;
    let mut score = 0.0;
    for (frame, &token) in path.iter().enumerate() {
        let probability = f64::from(rows[frame][token]);
        score += probability;
        if token == 0 {
            previous = None;
        } else if previous == Some(token) {
            spans.last_mut().unwrap().1 = frame + 1;
            let current: &mut f64 = log_probabilities.last_mut().unwrap();
            *current = current.max(probability);
        } else {
            tokens.push(token);
            spans.push((frame, frame + 1));
            log_probabilities.push(probability);
            previous = Some(token);
        }
    }
    (tokens, score, spans, log_probabilities)
}
