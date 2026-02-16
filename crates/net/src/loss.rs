use math::tensor::Tensor;

pub use crate::loss_functions::mse::MSE;

pub trait Loss {
    fn default() -> Self;
    fn forward(&self, inputs: Tensor, targets: Tensor) -> Tensor;
}
