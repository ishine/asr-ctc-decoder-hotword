use crate::DecoderError;

/// A borrowed row-major matrix of CTC log probabilities.
#[derive(Clone, Copy, Debug)]
pub struct LogProbabilities<'a> {
    values: &'a [f32],
    frames: usize,
    vocabulary_size: usize,
}

impl<'a> LogProbabilities<'a> {
    pub fn new(
        values: &'a [f32],
        frames: usize,
        vocabulary_size: usize,
    ) -> Result<Self, DecoderError> {
        if vocabulary_size == 0 {
            return Err(DecoderError::InvalidInput(
                "vocabulary_size must be at least 1",
            ));
        }
        if frames.checked_mul(vocabulary_size) != Some(values.len()) {
            return Err(DecoderError::InvalidInput(
                "values length must equal frames * vocabulary_size",
            ));
        }
        Ok(Self {
            values,
            frames,
            vocabulary_size,
        })
    }

    pub fn frames(self) -> usize {
        self.frames
    }

    pub fn vocabulary_size(self) -> usize {
        self.vocabulary_size
    }

    pub(crate) fn frame(self, frame: usize) -> &'a [f32] {
        let start = frame * self.vocabulary_size;
        &self.values[start..start + self.vocabulary_size]
    }
}
