use math::tensor::Tensor;
use crate::activation::Activation;
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Activation for Sigmoid {
    fn default() -> Self {
        Sigmoid
    }

    fn forward(&self, inputs: Tensor) -> Tensor {
        let output = inputs
            .iter()
            .map(|input| { 1.0 / (1.0 + (-*input).exp()) })
            .collect();

        Tensor::new(output, inputs.shape().to_vec())
    }
}
