use crate::context_policy::AdaptiveContextGate;
use crate::decoder::{DecodeMode, SearchConfig};
use crate::prefix_score::materialize_alignment;
use crate::{
    CTCDecoder, DecodeResult, DecoderError, GatingDiagnostics, HypothesisScore, LogProbabilities,
    PrefixBeamSearchOptions,
};

mod candidates;
mod transitions;

use candidates::{top_k, PreparedFrame};

impl CTCDecoder {
    pub fn prefix_beam_search(
        &mut self,
        input: LogProbabilities<'_>,
        options: PrefixBeamSearchOptions,
    ) -> Result<DecodeResult, DecoderError> {
        self.validate_input(input)?;
        let search_config = self.resolve_search_config(
            options.beam_size,
            options.token_beam_size,
            input.vocabulary_size(),
        )?;
        let frames = self.prepare_frames(input, search_config);
        self.lock_decode_mode(DecodeMode::PrefixBeam)?;
        if self.search_config.is_none() {
            self.search_config = Some(search_config);
        }

        for (frame_offset, frame) in frames.into_iter().enumerate() {
            self.processed_frames += 1;
            let (acoustic_scale, confidence_factor) = self.frame_gating(&frame.gating, frame.blank);
            let mut context_candidates = self.collect_context_candidates();
            let active_factor = AdaptiveContextGate::new(&self.context_policy.gating)
                .active_context_factor(
                    context_candidates.competing_count,
                    self.context_policy.max_injected_candidates,
                );
            let gating_factor = acoustic_scale * confidence_factor * active_factor;
            if !frame.gating.is_empty() && frame.blank < frame.values[0] {
                self.diagnostics.gating_factor_sum += gating_factor;
                self.diagnostics.nonblank_gating_frames += 1;
            }
            let context_threshold =
                self.context_policy.context_token_prune_threshold * gating_factor;
            let candidates = self.frame_candidates(
                input.frame(frame_offset),
                &frame,
                search_config,
                &mut context_candidates,
                context_threshold,
            );
            self.advance_prefix_frame(&candidates, search_config.beam_size);
        }

        let result = self.prefix_response(search_config, options);
        if options.finalize {
            self.reset();
        }
        Ok(result)
    }

    fn resolve_search_config(
        &self,
        beam_size: Option<usize>,
        token_beam_size: Option<usize>,
        vocabulary_size: usize,
    ) -> Result<SearchConfig, DecoderError> {
        if let Some(current) = self.search_config {
            if current.vocabulary_size != vocabulary_size {
                return Err(DecoderError::SearchConfigChanged("vocabulary_size"));
            }
            if beam_size.is_some_and(|value| value != current.beam_size) {
                return Err(DecoderError::SearchConfigChanged("beam_size"));
            }
            let requested = token_beam_size.map(|value| value.min(vocabulary_size));
            if requested.is_some_and(|value| value != current.token_beam_size) {
                return Err(DecoderError::SearchConfigChanged("token_beam_size"));
            }
            return Ok(current);
        }
        let (default_beam, default_token_beam, threshold) = self.decoding_quality.search_settings();
        let beam_size = beam_size.unwrap_or(default_beam);
        let token_beam_size = token_beam_size
            .unwrap_or(default_token_beam)
            .min(vocabulary_size);
        if beam_size == 0 {
            return Err(DecoderError::InvalidConfig("beam_size must be at least 1"));
        }
        if token_beam_size == 0 {
            return Err(DecoderError::InvalidConfig(
                "token_beam_size must be at least 1",
            ));
        }
        Ok(SearchConfig {
            beam_size,
            token_beam_size,
            token_prune_threshold: threshold,
            vocabulary_size,
        })
    }

    fn prepare_frames(
        &self,
        input: LogProbabilities<'_>,
        config: SearchConfig,
    ) -> Vec<PreparedFrame> {
        (0..input.frames())
            .map(|frame_index| {
                let row = input.frame(frame_index);
                let top = top_k(row, config.token_beam_size);
                let (indices, values): (Vec<_>, Vec<_>) = top.into_iter().unzip();
                let gating = if self.context_graph.is_some() && self.context_policy.gating.enabled {
                    top_k(row, row.len().min(8))
                        .into_iter()
                        .map(|(_, value)| value)
                        .collect()
                } else {
                    Vec::new()
                };
                PreparedFrame {
                    values,
                    indices,
                    blank: f64::from(row[self.blank_id]),
                    gating,
                }
            })
            .collect()
    }

    fn frame_gating(&mut self, top: &[f64], blank: f64) -> (f64, f64) {
        if top.is_empty() {
            return (1.0, 1.0);
        }
        let stats = AdaptiveContextGate::new(&self.context_policy.gating).analyze_frame(top, blank);
        self.diagnostics.gating_acoustic_scale_sum += stats.0;
        self.diagnostics.gating_frames += 1;
        stats
    }

    fn prefix_response(
        &self,
        search_config: SearchConfig,
        options: PrefixBeamSearchOptions,
    ) -> DecodeResult {
        let mut hypotheses = self.hypotheses.clone();
        hypotheses.sort_by(|left, right| right.1.total_score().total_cmp(&left.1.total_score()));
        let alignments: Vec<_> = hypotheses
            .iter()
            .map(|(_, score)| materialize_alignment(score.alignment()))
            .collect();
        DecodeResult {
            tokens: alignments
                .iter()
                .map(|alignment| alignment.0.clone())
                .collect(),
            timestamps: alignments
                .iter()
                .map(|alignment| self.timestamps(&alignment.1))
                .collect(),
            probabilities: options.return_token_probabilities.then(|| {
                alignments
                    .iter()
                    .map(|alignment| alignment.2.iter().map(|score| score.exp()).collect())
                    .collect()
            }),
            scores: options.return_scores.then(|| {
                hypotheses
                    .iter()
                    .map(|(_, score)| HypothesisScore {
                        acoustic: score.acoustic_score(),
                        contextual_bias: score.contextual_score(),
                        total: score.total_score(),
                    })
                    .collect()
            }),
            gating: options
                .return_gating_diagnostics
                .then(|| self.gating_diagnostics(search_config)),
        }
    }

    fn gating_diagnostics(&self, config: SearchConfig) -> GatingDiagnostics {
        let enabled = self.context_graph.is_some() && self.context_policy.gating.enabled;
        GatingDiagnostics {
            contextual_biasing: self.context_graph.is_some(),
            adaptive_context_gating: enabled,
            gating_frames: if enabled {
                self.diagnostics.gating_frames
            } else {
                0
            },
            mean_gating_acoustic_scale: (enabled && self.diagnostics.gating_frames > 0).then(
                || {
                    self.diagnostics.gating_acoustic_scale_sum
                        / self.diagnostics.gating_frames as f64
                },
            ),
            mean_gating_factor: (enabled && self.diagnostics.nonblank_gating_frames > 0).then(
                || {
                    self.diagnostics.gating_factor_sum
                        / self.diagnostics.nonblank_gating_frames as f64
                },
            ),
            nonblank_gating_frames: if enabled {
                self.diagnostics.nonblank_gating_frames
            } else {
                0
            },
            evaluated_context_candidates: self.diagnostics.evaluated_context_candidates,
            gathered_context_candidates: self.diagnostics.gathered_context_candidates,
            max_context_candidates_per_frame: self.diagnostics.max_context_candidates_per_frame,
            context_candidate_limit_hits: self.diagnostics.context_candidate_limit_hits,
            word_boundary_aware_context: !self.word_boundary_token_ids.is_empty(),
            boundary_suppressed_context_candidates: self
                .diagnostics
                .boundary_suppressed_context_candidates,
            beam_size: config.beam_size,
            token_beam_size: config.token_beam_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContextPolicy, DecoderConfig};

    #[test]
    fn streaming_matches_one_shot() {
        let values = [-5.0, 0.0, -5.0, 0.0, -5.0, -5.0, -5.0, -5.0, 0.0];
        let all = LogProbabilities::new(&values, 3, 3).unwrap();
        let first = LogProbabilities::new(&values[..6], 2, 3).unwrap();
        let second = LogProbabilities::new(&values[6..], 1, 3).unwrap();
        let mut one_shot = CTCDecoder::new(DecoderConfig::default()).unwrap();
        let expected = one_shot
            .prefix_beam_search(
                all,
                PrefixBeamSearchOptions {
                    finalize: true,
                    ..PrefixBeamSearchOptions::default()
                },
            )
            .unwrap();
        let mut streaming = CTCDecoder::new(DecoderConfig::default()).unwrap();
        streaming
            .prefix_beam_search(first, PrefixBeamSearchOptions::default())
            .unwrap();
        let actual = streaming
            .prefix_beam_search(
                second,
                PrefixBeamSearchOptions {
                    finalize: true,
                    ..PrefixBeamSearchOptions::default()
                },
            )
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn context_completion_changes_ranking() {
        let values = [-4.0, -0.1, -2.0, -4.0, -2.0, -0.2];
        let input = LogProbabilities::new(&values, 2, 3).unwrap();
        let mut decoder = CTCDecoder::new(DecoderConfig {
            context_token_ids: vec![vec![1, 2]],
            context_policy: Some(ContextPolicy {
                completion_bonus: 5.0,
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
                    ..PrefixBeamSearchOptions::default()
                },
            )
            .unwrap();
        assert_eq!(result.tokens[0], vec![1, 2]);
        assert_eq!(result.scores.unwrap()[0].contextual_bias, 5.0);
    }
}
