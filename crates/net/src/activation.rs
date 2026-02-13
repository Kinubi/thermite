use math::tensor::Tensor;

pub use crate::activations::Step;

pub trait Activation {
    fn new() -> Self;
    fn forward(&self, inputs: Tensor) -> Tensor;
}
