use std::collections::{HashMap, HashSet, VecDeque};

use crate::ContextPolicy;

#[derive(Clone, Debug)]
struct ContextState {
    transitions: HashMap<usize, usize>,
    transition_order: Vec<usize>,
    failure: usize,
    own_match: Option<(usize, f64)>,
    output: Option<usize>,
}

impl ContextState {
    fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            transition_order: Vec::new(),
            failure: 0,
            own_match: None,
            output: None,
        }
    }
}

pub(crate) struct ActiveContextCandidates {
    pub tokens: Vec<usize>,
    pub competing_count: usize,
    pub truncated: bool,
}

/// A compact Aho-Corasick automaton over model token IDs.
#[derive(Clone, Debug)]
pub(crate) struct ContextGraph {
    states: Vec<ContextState>,
    max_completed_contexts_per_token: usize,
    pub max_token_id: usize,
}

impl ContextGraph {
    pub fn from_token_ids(contexts: &[Vec<usize>], policy: &ContextPolicy) -> Option<Self> {
        let mut graph = Self {
            states: vec![ContextState::new()],
            max_completed_contexts_per_token: policy.max_completed_contexts_per_token,
            max_token_id: 0,
        };
        let mut seen = HashSet::new();
        for tokens in contexts.iter().filter(|tokens| !tokens.is_empty()) {
            if !seen.insert(tokens.clone()) {
                continue;
            }
            let mut state = 0;
            for &token in tokens {
                graph.max_token_id = graph.max_token_id.max(token);
                let next = if let Some(&next) = graph.states[state].transitions.get(&token) {
                    next
                } else {
                    let next = graph.states.len();
                    graph.states.push(ContextState::new());
                    graph.states[state].transitions.insert(token, next);
                    graph.states[state].transition_order.push(token);
                    next
                };
                state = next;
            }
            graph.states[state].own_match = Some((tokens.len(), policy.completion_bonus));
        }
        if graph.states.len() == 1 {
            return None;
        }
        graph.fill_failure_and_matches();
        Some(graph)
    }

    fn fill_failure_and_matches(&mut self) {
        let mut queue = VecDeque::new();
        let root_children: Vec<_> = self.states[0].transitions.values().copied().collect();
        for child in root_children {
            self.states[child].failure = 0;
            queue.push_back(child);
        }
        while let Some(current) = queue.pop_front() {
            let transitions: Vec<_> = self.states[current]
                .transitions
                .iter()
                .map(|(&token, &state)| (token, state))
                .collect();
            for (token, next) in transitions {
                let mut failure = self.states[current].failure;
                while failure != 0 && !self.states[failure].transitions.contains_key(&token) {
                    failure = self.states[failure].failure;
                }
                if let Some(&target) = self.states[failure].transitions.get(&token) {
                    failure = target;
                }
                self.states[next].failure = failure;
                self.states[next].output = if self.states[failure].own_match.is_some() {
                    Some(failure)
                } else {
                    self.states[failure].output
                };
                queue.push_back(next);
            }
        }
    }

    pub fn forward_one_step(&self, mut state: usize, token: usize) -> usize {
        while state != 0 && !self.states[state].transitions.contains_key(&token) {
            state = self.states[state].failure;
        }
        self.states[state]
            .transitions
            .get(&token)
            .copied()
            .unwrap_or(state)
    }

    pub fn completed_matches(&self, mut state: usize) -> Vec<(usize, f64)> {
        let mut matches = Vec::new();
        if let Some(found) = self.states[state].own_match {
            matches.push(found);
        }
        while matches.len() < self.max_completed_contexts_per_token {
            let Some(output) = self.states[state].output else {
                break;
            };
            state = output;
            if let Some(found) = self.states[state].own_match {
                matches.push(found);
            }
        }
        matches
    }

    pub fn active_candidates(&self, mut state: usize, maximum: usize) -> ActiveContextCandidates {
        if maximum == 0 {
            return ActiveContextCandidates {
                tokens: Vec::new(),
                competing_count: 0,
                truncated: !self.states[state].transitions.is_empty(),
            };
        }
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        let mut competing_count = 0;
        loop {
            let current = &self.states[state];
            competing_count = (competing_count + current.transitions.len()).min(maximum + 1);
            if state == 0 && current.transitions.len() > maximum - candidates.len() {
                return ActiveContextCandidates {
                    tokens: candidates,
                    competing_count,
                    truncated: true,
                };
            }
            for &token in &current.transition_order {
                if seen.insert(token) {
                    candidates.push(token);
                    if candidates.len() == maximum {
                        return ActiveContextCandidates {
                            tokens: candidates,
                            competing_count,
                            truncated: state != 0
                                || current
                                    .transition_order
                                    .iter()
                                    .any(|token| !seen.contains(token)),
                        };
                    }
                }
            }
            if state == 0 {
                break;
            }
            state = current.failure;
        }
        ActiveContextCandidates {
            tokens: candidates,
            competing_count,
            truncated: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_overlapping_suffix_matches() {
        let policy = ContextPolicy::conservative();
        let graph = ContextGraph::from_token_ids(&[vec![1, 2], vec![2]], &policy).unwrap();
        let state = graph.forward_one_step(graph.forward_one_step(0, 1), 2);
        assert_eq!(graph.completed_matches(state), vec![(2, 3.5), (1, 3.5)]);
    }

    #[test]
    fn large_root_is_not_materialized() {
        let contexts: Vec<_> = (1..=1000).map(|token| vec![token]).collect();
        let graph =
            ContextGraph::from_token_ids(&contexts, &ContextPolicy::conservative()).unwrap();
        let active = graph.active_candidates(0, 12);
        assert!(active.tokens.is_empty());
        assert!(active.truncated);
        assert_eq!(active.competing_count, 13);
    }
}
