use std::error::Error;
use std::fmt::{Display, Formatter};

/// Validation and stream-lifecycle errors returned by the decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderError {
    InvalidConfig(&'static str),
    InvalidInput(&'static str),
    DecodeModeChanged,
    SearchConfigChanged(&'static str),
}

impl Display for DecoderError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) | Self::InvalidInput(message) => {
                formatter.write_str(message)
            }
            Self::DecodeModeChanged => formatter
                .write_str("cannot switch search method within a stream; finalize or reset first"),
            Self::SearchConfigChanged(name) => {
                write!(formatter, "{name} cannot change within a stream")
            }
        }
    }
}

impl Error for DecoderError {}
