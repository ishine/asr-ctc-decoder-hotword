/// A zero-based, half-open token span.
#[derive(Clone, Debug, PartialEq)]
pub struct Timestamp {
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_ms: Option<f64>,
    pub end_ms: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HypothesisScore {
    pub acoustic: f64,
    pub contextual_bias: f64,
    pub total: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GatingDiagnostics {
    pub contextual_biasing: bool,
    pub adaptive_context_gating: bool,
    pub gating_frames: usize,
    pub mean_gating_acoustic_scale: Option<f64>,
    pub mean_gating_factor: Option<f64>,
    pub nonblank_gating_frames: usize,
    pub evaluated_context_candidates: usize,
    pub gathered_context_candidates: usize,
    pub max_context_candidates_per_frame: usize,
    pub context_candidate_limit_hits: usize,
    pub word_boundary_aware_context: bool,
    pub boundary_suppressed_context_candidates: usize,
    pub beam_size: usize,
    pub token_beam_size: usize,
}

/// Search output. Greedy search returns one hypothesis; beam search returns N-best.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeResult {
    pub tokens: Vec<Vec<usize>>,
    pub timestamps: Vec<Vec<Timestamp>>,
    pub probabilities: Option<Vec<Vec<f64>>>,
    pub scores: Option<Vec<HypothesisScore>>,
    pub gating: Option<GatingDiagnostics>,
}
