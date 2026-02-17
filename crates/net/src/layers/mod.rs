pub mod dense;
pub mod activations;

pub use dense::Linear;
pub use activations::{ Linear as LinearActivation, ReLU, Sigmoid, Softmax, Step };
