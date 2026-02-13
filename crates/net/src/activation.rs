use math::tensor::Tensor;

pub use crate::activations::Step;
pub use crate::activations::Linear;
pub use crate::activations::Sigmoid;
pub use crate::activations::ReLU;
pub use crate::activations::Softmax;

pub trait Activation {
    fn default() -> Self;
    fn forward(&self, inputs: Tensor) -> Tensor;
}
