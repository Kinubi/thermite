use math::tensor::Tensor;
use crate::activation::Activation;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Activation for ReLU {
    fn default() -> Self {
        ReLU
    }

    fn forward(&self, inputs: Tensor) -> Tensor {
        let output = inputs
            .iter()
            .map(|input| { input.max(0.0) })
            .collect();

        Tensor::new(output, inputs.shape().to_vec())
    }
}
