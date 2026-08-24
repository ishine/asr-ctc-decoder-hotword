use std::rc::Rc;

#[derive(Clone, Debug)]
pub(crate) struct AlignmentNode {
    pub parent: Option<Rc<Self>>,
    pub token_id: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub log_probability: f64,
    pub length: usize,
}

pub(crate) type Alignment = Option<Rc<AlignmentNode>>;

pub(crate) fn append_alignment(
    alignment: &Alignment,
    token_id: usize,
    frame: usize,
    log_probability: f64,
) -> Alignment {
    Some(Rc::new(AlignmentNode {
        parent: alignment.clone(),
        token_id,
        start_frame: frame,
        end_frame: frame + 1,
        log_probability,
        length: alignment.as_ref().map_or(1, |node| node.length + 1),
    }))
}

pub(crate) fn extend_alignment(
    alignment: &Alignment,
    end_frame: usize,
    log_probability: f64,
) -> Alignment {
    alignment.as_ref().map(|node| {
        Rc::new(AlignmentNode {
            parent: node.parent.clone(),
            token_id: node.token_id,
            start_frame: node.start_frame,
            end_frame,
            log_probability: node.log_probability.max(log_probability),
            length: node.length,
        })
    })
}

pub(crate) type MaterializedAlignment = (Vec<usize>, Vec<(usize, usize)>, Vec<f64>);

pub(crate) fn materialize_alignment(alignment: &Alignment) -> MaterializedAlignment {
    let Some(tail) = alignment else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let mut tokens = vec![0; tail.length];
    let mut spans = vec![(0, 0); tail.length];
    let mut log_probabilities = vec![0.0; tail.length];
    let mut current = Some(tail.clone());
    let mut position = tail.length;
    while let Some(node) = current {
        position -= 1;
        tokens[position] = node.token_id;
        spans[position] = (node.start_frame, node.end_frame);
        log_probabilities[position] = node.log_probability;
        current = node.parent.clone();
    }
    (tokens, spans, log_probabilities)
}

#[derive(Debug)]
struct ContextScoreNode {
    jumps: Vec<Rc<Self>>,
    length: usize,
    best_score: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ContextScoreState(Rc<ContextScoreNode>);

impl Default for ContextScoreState {
    fn default() -> Self {
        Self(Rc::new(ContextScoreNode {
            jumps: Vec::new(),
            length: 0,
            best_score: 0.0,
        }))
    }
}

impl ContextScoreState {
    fn ancestor(&self, mut distance: usize) -> Rc<ContextScoreNode> {
        debug_assert!(distance <= self.0.length);
        let mut node = self.0.clone();
        let mut level = 0;
        while distance > 0 {
            if distance & 1 == 1 {
                node = node.jumps[level].clone();
            }
            distance >>= 1;
            level += 1;
        }
        node
    }

    pub fn advance(&self, matches: &[(usize, f64)]) -> Self {
        let mut best_score = self.0.best_score;
        for &(token_count, bonus) in matches {
            let predecessor = self.ancestor(token_count - 1);
            best_score = best_score.max(predecessor.best_score + bonus);
        }
        let parent = self.0.clone();
        let mut jumps = vec![parent.clone()];
        let mut level = 1;
        while level - 1 < jumps[level - 1].jumps.len() {
            let next = jumps[level - 1].jumps[level - 1].clone();
            jumps.push(next);
            level += 1;
        }
        Self(Rc::new(ContextScoreNode {
            jumps,
            length: self.0.length + 1,
            best_score,
        }))
    }

    pub fn best_score(&self) -> f64 {
        self.0.best_score
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PrefixScore {
    pub blank_score: f64,
    pub non_blank_score: f64,
    pub viterbi_blank_score: f64,
    pub viterbi_non_blank_score: f64,
    pub blank_alignment: Alignment,
    pub non_blank_alignment: Alignment,
    pub context_state: usize,
    pub context_score_state: ContextScoreState,
    pub has_context: bool,
}

impl Default for PrefixScore {
    fn default() -> Self {
        Self {
            blank_score: f64::NEG_INFINITY,
            non_blank_score: f64::NEG_INFINITY,
            viterbi_blank_score: f64::NEG_INFINITY,
            viterbi_non_blank_score: f64::NEG_INFINITY,
            blank_alignment: None,
            non_blank_alignment: None,
            context_state: 0,
            context_score_state: ContextScoreState::default(),
            has_context: false,
        }
    }
}

impl PrefixScore {
    pub fn initial() -> Self {
        Self {
            blank_score: 0.0,
            viterbi_blank_score: 0.0,
            ..Self::default()
        }
    }

    pub fn acoustic_score(&self) -> f64 {
        log_add(self.blank_score, self.non_blank_score)
    }

    pub fn viterbi_score(&self) -> f64 {
        self.viterbi_blank_score.max(self.viterbi_non_blank_score)
    }

    pub fn alignment(&self) -> &Alignment {
        if self.viterbi_blank_score > self.viterbi_non_blank_score {
            &self.blank_alignment
        } else {
            &self.non_blank_alignment
        }
    }

    pub fn contextual_score(&self) -> f64 {
        self.context_score_state.best_score()
    }

    pub fn total_score(&self) -> f64 {
        self.acoustic_score() + self.contextual_score()
    }
}

pub(crate) fn log_add(first: f64, second: f64) -> f64 {
    if first == f64::NEG_INFINITY {
        return second;
    }
    if second == f64::NEG_INFINITY {
        return first;
    }
    let maximum = first.max(second);
    maximum + ((first - maximum).exp() + (second - maximum).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_alignment_replaces_only_tail() {
        let first = append_alignment(&None, 1, 0, -2.0);
        let second = append_alignment(&first, 2, 1, -3.0);
        let extended = extend_alignment(&second, 3, -1.0);
        assert_eq!(
            materialize_alignment(&extended),
            (vec![1, 2], vec![(0, 1), (1, 3)], vec![-2.0, -1.0])
        );
    }
}
