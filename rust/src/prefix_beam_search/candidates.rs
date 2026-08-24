use std::collections::{BTreeMap, HashSet};

use crate::decoder::SearchConfig;
use crate::CTCDecoder;

pub(super) struct ContextCandidates {
    pub tokens: Vec<usize>,
    pub root_tokens: HashSet<usize>,
    pub non_root_tokens: HashSet<usize>,
    pub competing_count: usize,
}

impl ContextCandidates {
    fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            root_tokens: HashSet::new(),
            non_root_tokens: HashSet::new(),
            competing_count: 0,
        }
    }
}

pub(super) struct PreparedFrame {
    pub values: Vec<f64>,
    pub indices: Vec<usize>,
    pub blank: f64,
    pub gating: Vec<f64>,
}

impl CTCDecoder {
    pub(super) fn collect_context_candidates(&mut self) -> ContextCandidates {
        let maximum = self.context_policy.max_injected_candidates;
        let Some(graph) = self.context_graph.as_ref() else {
            return ContextCandidates::empty();
        };
        if maximum == 0 {
            return ContextCandidates::empty();
        }

        let mut tokens = Vec::new();
        let mut token_set = HashSet::new();
        let mut root_tokens = HashSet::new();
        let mut non_root_tokens = HashSet::new();
        let mut seen_states = HashSet::new();
        let mut competing_count = 0;
        let mut truncated = false;
        for (_, score) in &self.hypotheses {
            if !seen_states.insert(score.context_state) {
                continue;
            }
            let active = graph.active_candidates(score.context_state, maximum);
            competing_count = (competing_count + active.competing_count).min(maximum + 1);
            truncated |= active.truncated;
            let target = if score.context_state == 0 {
                &mut root_tokens
            } else {
                &mut non_root_tokens
            };
            target.extend(active.tokens.iter().copied());
            for token_id in active.tokens {
                if !token_set.insert(token_id) {
                    continue;
                }
                if tokens.len() == maximum {
                    truncated = true;
                    break;
                }
                tokens.push(token_id);
            }
        }
        if truncated {
            self.diagnostics.context_candidate_limit_hits += 1;
        }
        self.diagnostics.max_context_candidates_per_frame = self
            .diagnostics
            .max_context_candidates_per_frame
            .max(tokens.len());
        self.diagnostics.evaluated_context_candidates += tokens.len();
        ContextCandidates {
            tokens,
            root_tokens,
            non_root_tokens,
            competing_count,
        }
    }

    pub(super) fn frame_candidates(
        &mut self,
        row: &[f32],
        frame: &PreparedFrame,
        config: SearchConfig,
        context: &mut ContextCandidates,
        context_threshold: f64,
    ) -> Vec<(f64, usize)> {
        let blocked = self.apply_word_boundary_guard(frame, context, context_threshold);
        let mut scores: BTreeMap<_, _> = frame
            .indices
            .iter()
            .copied()
            .zip(frame.values.iter().copied())
            .collect();
        scores.insert(self.blank_id, frame.blank);
        for token in blocked {
            scores.remove(&token);
        }
        let missing: Vec<_> = context
            .tokens
            .iter()
            .copied()
            .filter(|token| !scores.contains_key(token))
            .collect();
        self.diagnostics.gathered_context_candidates += missing.len();
        for token in missing {
            let probability = f64::from(row[token]);
            if probability >= frame.values[0] - context_threshold {
                scores.insert(token, probability);
            }
        }
        let mut candidates: Vec<_> = scores
            .into_iter()
            .filter_map(|(token, probability)| {
                (probability >= frame.values[0] - config.token_prune_threshold
                    || token == self.blank_id)
                    .then_some((probability, token))
            })
            .collect();
        candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| right.1.cmp(&left.1))
        });
        candidates
    }

    fn apply_word_boundary_guard(
        &mut self,
        frame: &PreparedFrame,
        context: &mut ContextCandidates,
        threshold: f64,
    ) -> HashSet<usize> {
        if self.word_boundary_token_ids.is_empty() {
            return HashSet::new();
        }
        let boundary_best = frame
            .indices
            .iter()
            .zip(&frame.values)
            .filter(|(token, _)| self.word_boundary_token_ids.contains(token))
            .map(|(_, &value)| value)
            .max_by(f64::total_cmp)
            .unwrap_or(f64::NEG_INFINITY);
        if boundary_best < frame.values[0] - threshold {
            return HashSet::new();
        }
        let acoustic_tokens: HashSet<_> = frame.indices.iter().copied().collect();
        let blocked: HashSet<_> = context
            .non_root_tokens
            .difference(&context.root_tokens)
            .copied()
            .filter(|token| !self.word_boundary_token_ids.contains(token))
            .filter(|token| !acoustic_tokens.contains(token))
            .collect();
        if !blocked.is_empty() {
            context.tokens.retain(|token| !blocked.contains(token));
            self.diagnostics.boundary_suppressed_context_candidates += blocked.len();
        }
        blocked
    }
}

pub(super) fn top_k(row: &[f32], count: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<_> = row
        .iter()
        .enumerate()
        .map(|(token, &probability)| (token, f64::from(probability)))
        .collect();
    indexed.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.0.cmp(&left.0))
    });
    indexed.truncate(count);
    indexed
}
