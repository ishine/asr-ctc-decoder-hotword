use std::collections::BTreeMap;

use crate::context_graph::ContextGraph;
use crate::prefix_score::{append_alignment, extend_alignment, log_add, PrefixScore};
use crate::CTCDecoder;

impl CTCDecoder {
    pub(super) fn advance_prefix_frame(&mut self, candidates: &[(f64, usize)], beam_size: usize) {
        let previous = std::mem::take(&mut self.hypotheses);
        let graph = self.context_graph.as_ref();
        let mut next: BTreeMap<Vec<usize>, PrefixScore> = BTreeMap::new();
        for &(probability, token_id) in candidates {
            for (prefix, prefix_score) in &previous {
                if token_id == self.blank_id {
                    update_blank(
                        &mut next,
                        prefix,
                        prefix_score,
                        probability,
                        graph.is_some(),
                    );
                } else if prefix.last() == Some(&token_id) {
                    update_repeated(
                        &mut next,
                        prefix,
                        prefix_score,
                        token_id,
                        probability,
                        self.processed_frames,
                        graph,
                    );
                } else {
                    update_extended(
                        &mut next,
                        prefix,
                        prefix_score,
                        token_id,
                        probability,
                        self.processed_frames,
                        graph,
                    );
                }
            }
        }
        let mut ranked: Vec<_> = next
            .into_iter()
            .filter(|(_, score)| score.acoustic_score().is_finite())
            .collect();
        ranked.sort_by(|left, right| right.1.total_score().total_cmp(&left.1.total_score()));
        ranked.truncate(beam_size);
        self.hypotheses = ranked;
    }
}

fn copy_context_state(source: &PrefixScore, destination: &mut PrefixScore, enabled: bool) {
    if enabled && !destination.has_context {
        destination.context_state = source.context_state;
        destination.context_score_state = source.context_score_state.clone();
        destination.has_context = true;
    }
}

fn advance_context(
    graph: Option<&ContextGraph>,
    source: &PrefixScore,
    destination: &mut PrefixScore,
    token_id: usize,
) {
    let Some(graph) = graph else {
        return;
    };
    let context_state = graph.forward_one_step(source.context_state, token_id);
    let context_score_state = source
        .context_score_state
        .advance(&graph.completed_matches(context_state));
    if !destination.has_context
        || context_score_state.best_score() > destination.context_score_state.best_score()
    {
        destination.context_state = context_state;
        destination.context_score_state = context_score_state;
        destination.has_context = true;
    }
}

fn update_blank(
    next: &mut BTreeMap<Vec<usize>, PrefixScore>,
    prefix: &[usize],
    source: &PrefixScore,
    probability: f64,
    context_enabled: bool,
) {
    let destination = next.entry(prefix.to_vec()).or_default();
    destination.blank_score = log_add(
        destination.blank_score,
        source.acoustic_score() + probability,
    );
    let viterbi = source.viterbi_score() + probability;
    if viterbi > destination.viterbi_blank_score {
        destination.viterbi_blank_score = viterbi;
        destination.blank_alignment = source.alignment().clone();
    }
    copy_context_state(source, destination, context_enabled);
}

fn update_repeated(
    next: &mut BTreeMap<Vec<usize>, PrefixScore>,
    prefix: &[usize],
    source: &PrefixScore,
    token_id: usize,
    probability: f64,
    processed_frames: usize,
    graph: Option<&ContextGraph>,
) {
    let repeated = next.entry(prefix.to_vec()).or_default();
    repeated.non_blank_score = log_add(
        repeated.non_blank_score,
        source.non_blank_score + probability,
    );
    if repeated.viterbi_non_blank_score < source.viterbi_non_blank_score + probability {
        repeated.viterbi_non_blank_score = source.viterbi_non_blank_score + probability;
        repeated.non_blank_alignment =
            extend_alignment(&source.non_blank_alignment, processed_frames, probability);
    }
    copy_context_state(source, repeated, graph.is_some());

    let mut extended_prefix = prefix.to_vec();
    extended_prefix.push(token_id);
    let after_blank = next.entry(extended_prefix).or_default();
    after_blank.non_blank_score = log_add(
        after_blank.non_blank_score,
        source.blank_score + probability,
    );
    if after_blank.viterbi_non_blank_score < source.viterbi_blank_score + probability {
        after_blank.viterbi_non_blank_score = source.viterbi_blank_score + probability;
        after_blank.non_blank_alignment = append_alignment(
            &source.blank_alignment,
            token_id,
            processed_frames - 1,
            probability,
        );
    }
    advance_context(graph, source, after_blank, token_id);
}

fn update_extended(
    next: &mut BTreeMap<Vec<usize>, PrefixScore>,
    prefix: &[usize],
    source: &PrefixScore,
    token_id: usize,
    probability: f64,
    processed_frames: usize,
    graph: Option<&ContextGraph>,
) {
    let mut extended_prefix = prefix.to_vec();
    extended_prefix.push(token_id);
    let destination = next.entry(extended_prefix).or_default();
    destination.non_blank_score = log_add(
        destination.non_blank_score,
        source.acoustic_score() + probability,
    );
    if destination.viterbi_non_blank_score < source.viterbi_score() + probability {
        destination.viterbi_non_blank_score = source.viterbi_score() + probability;
        destination.non_blank_alignment = append_alignment(
            source.alignment(),
            token_id,
            processed_frames - 1,
            probability,
        );
    }
    advance_context(graph, source, destination, token_id);
}
