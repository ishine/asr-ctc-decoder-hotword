use crate::decoder::DecodeMode;
use crate::{CTCDecoder, DecodeResult, DecoderError, LogProbabilities};

impl CTCDecoder {
    pub fn greedy_search(
        &mut self,
        input: LogProbabilities<'_>,
        finalize: bool,
        return_token_probabilities: bool,
    ) -> Result<DecodeResult, DecoderError> {
        self.validate_input(input)?;
        self.lock_decode_mode(DecodeMode::Greedy)?;
        for frame_index in 0..input.frames() {
            let row = input.frame(frame_index);
            let mut token_id = 0;
            for candidate in 1..row.len() {
                if row[candidate] > row[token_id] {
                    token_id = candidate;
                }
            }
            let score = f64::from(row[token_id]);
            self.consume_greedy_frame(token_id, score);
        }

        let result = DecodeResult {
            tokens: vec![self.greedy_tokens.clone()],
            timestamps: vec![self.timestamps(&self.greedy_spans)],
            probabilities: return_token_probabilities.then(|| {
                vec![self
                    .greedy_token_log_probabilities
                    .iter()
                    .map(|score| score.exp())
                    .collect()]
            }),
            scores: None,
            gating: None,
        };
        if finalize {
            self.reset();
        }
        Ok(result)
    }

    fn consume_greedy_frame(&mut self, token_id: usize, score: f64) {
        self.processed_frames += 1;
        if token_id == self.blank_id {
            self.last_greedy_token = None;
            return;
        }
        if self.last_greedy_token == Some(token_id) {
            let span = self
                .greedy_spans
                .last_mut()
                .expect("a repeated token has an existing span");
            span.1 = self.processed_frames;
            let probability = self
                .greedy_token_log_probabilities
                .last_mut()
                .expect("a repeated token has an existing probability");
            *probability = probability.max(score);
            return;
        }
        self.greedy_tokens.push(token_id);
        self.greedy_spans
            .push((self.processed_frames - 1, self.processed_frames));
        self.greedy_token_log_probabilities.push(score);
        self.last_greedy_token = Some(token_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecoderConfig;

    #[test]
    fn collapses_repeats_and_blank_separated_tokens() {
        let values = [
            -5.0, 0.0, -5.0, -5.0, 0.0, -5.0, 0.0, -5.0, -5.0, -5.0, 0.0, -5.0,
        ];
        let input = LogProbabilities::new(&values, 4, 3).unwrap();
        let mut decoder = CTCDecoder::new(DecoderConfig {
            frame_shift_ms: Some(20.0),
            ..DecoderConfig::default()
        })
        .unwrap();
        let result = decoder.greedy_search(input, true, true).unwrap();
        assert_eq!(result.tokens, vec![vec![1, 1]]);
        assert_eq!(result.timestamps[0][0].end_frame, 2);
        assert_eq!(result.timestamps[0][1].start_ms, Some(60.0));
    }
}
