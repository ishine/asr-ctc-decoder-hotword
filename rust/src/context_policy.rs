use crate::DecoderError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HotwordStrength {
    Conservative,
    #[default]
    Balanced,
    Aggressive,
    LowFalseActivation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodingQuality {
    LowLatency,
    #[default]
    Balanced,
    HighAccuracy,
}

impl DecodingQuality {
    pub(crate) fn search_settings(self) -> (usize, usize, f64) {
        match self {
            Self::LowLatency => (4, 8, 3.0),
            Self::Balanced => (8, 16, 4.0),
            Self::HighAccuracy => (12, 24, 5.0),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveGatingConfig {
    pub enabled: bool,
    pub reference_top_spread: f64,
    pub min_acoustic_scale: f64,
    pub max_acoustic_scale: f64,
    pub min_entropy_confidence_factor: f64,
    pub max_active_context_penalty: f64,
}

impl Default for AdaptiveGatingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reference_top_spread: 4.0,
            min_acoustic_scale: 0.5,
            max_acoustic_scale: 2.0,
            min_entropy_confidence_factor: 0.65,
            max_active_context_penalty: 0.35,
        }
    }
}

impl AdaptiveGatingConfig {
    pub(crate) fn validate(&self) -> Result<(), DecoderError> {
        if !finite_positive(self.reference_top_spread)
            || !finite_positive(self.min_acoustic_scale)
            || !finite_positive(self.max_acoustic_scale)
        {
            return Err(DecoderError::InvalidConfig(
                "acoustic gating scales must be finite and positive",
            ));
        }
        if self.min_acoustic_scale > self.max_acoustic_scale {
            return Err(DecoderError::InvalidConfig(
                "acoustic scale limits must be ordered",
            ));
        }
        if !finite_open_closed(self.min_entropy_confidence_factor, 0.0, 1.0)
            || !finite_closed_open(self.max_active_context_penalty, 0.0, 1.0)
        {
            return Err(DecoderError::InvalidConfig(
                "gating factor is outside its valid range",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextPolicy {
    pub completion_bonus: f64,
    pub context_token_prune_threshold: f64,
    pub max_injected_candidates: usize,
    pub max_completed_contexts_per_token: usize,
    pub gating: AdaptiveGatingConfig,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self::conservative()
    }
}

impl ContextPolicy {
    pub fn conservative() -> Self {
        Self {
            completion_bonus: 3.5,
            context_token_prune_threshold: 1.5,
            max_injected_candidates: 8,
            max_completed_contexts_per_token: 8,
            gating: AdaptiveGatingConfig::default(),
        }
    }

    pub fn balanced() -> Self {
        Self {
            completion_bonus: 5.0,
            context_token_prune_threshold: 2.5,
            max_injected_candidates: 12,
            ..Self::conservative()
        }
    }

    pub fn aggressive() -> Self {
        Self {
            completion_bonus: 7.0,
            context_token_prune_threshold: 3.5,
            max_injected_candidates: 16,
            ..Self::conservative()
        }
    }

    pub fn low_false_activation() -> Self {
        Self {
            completion_bonus: 1.0,
            context_token_prune_threshold: 0.5,
            max_injected_candidates: 8,
            ..Self::conservative()
        }
    }

    pub(crate) fn for_strength(strength: HotwordStrength) -> Self {
        match strength {
            HotwordStrength::Conservative => Self::conservative(),
            HotwordStrength::Balanced => Self::balanced(),
            HotwordStrength::Aggressive => Self::aggressive(),
            HotwordStrength::LowFalseActivation => Self::low_false_activation(),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), DecoderError> {
        if !finite_non_negative(self.completion_bonus)
            || !finite_non_negative(self.context_token_prune_threshold)
        {
            return Err(DecoderError::InvalidConfig(
                "context scores must be finite and non-negative",
            ));
        }
        if self.max_completed_contexts_per_token == 0 {
            return Err(DecoderError::InvalidConfig(
                "max_completed_contexts_per_token must be positive",
            ));
        }
        self.gating.validate()
    }
}

pub(crate) struct AdaptiveContextGate<'a> {
    config: &'a AdaptiveGatingConfig,
}

impl<'a> AdaptiveContextGate<'a> {
    pub(crate) fn new(config: &'a AdaptiveGatingConfig) -> Self {
        Self { config }
    }

    pub(crate) fn analyze_frame(&self, top: &[f64], blank: f64) -> (f64, f64) {
        if !self.config.enabled || top.len() < 2 {
            return (1.0, 1.0);
        }
        let finite: Vec<_> = top
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if finite.len() < 2 {
            return (1.0, 1.0);
        }
        let best = top[0];
        let spread = best - finite[finite.len() - 1];
        let acoustic_scale = (spread / self.config.reference_top_spread).clamp(
            self.config.min_acoustic_scale,
            self.config.max_acoustic_scale,
        );
        let probabilities: Vec<_> = finite.iter().map(|value| (value - best).exp()).collect();
        let total: f64 = probabilities.iter().sum();
        let entropy = probabilities
            .iter()
            .map(|probability| probability / total)
            .filter(|probability| *probability > 0.0)
            .map(|probability| -probability * probability.ln())
            .sum::<f64>();
        let normalized_entropy = entropy / (finite.len() as f64).ln();
        let mut confidence =
            1.0 - (1.0 - self.config.min_entropy_confidence_factor) * normalized_entropy;
        if blank == best {
            let blank_margin = (best - finite[1]) / acoustic_scale.max(1e-6);
            if blank_margin > 1.0 {
                confidence *= (1.0 / blank_margin).max(0.25);
            }
        }
        (acoustic_scale, confidence)
    }

    pub(crate) fn active_context_factor(&self, count: usize, maximum: usize) -> f64 {
        if !self.config.enabled || count <= 1 || maximum <= 1 {
            return 1.0;
        }
        let pressure = ((count - 1) as f64).ln_1p() / (maximum as f64).ln();
        1.0 - self.config.max_active_context_penalty * pressure.min(1.0)
    }
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn finite_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn finite_open_closed(value: f64, lower: f64, upper: f64) -> bool {
    value.is_finite() && value > lower && value <= upper
}

fn finite_closed_open(value: f64, lower: f64, upper: f64) -> bool {
    value.is_finite() && value >= lower && value < upper
}
