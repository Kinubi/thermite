use math::tensor::Tensor;
use crate::activation::Activation;
pub struct Linear;

impl Linear {
    pub fn new() -> Self {
        Linear
    }
}

impl Activation for Linear {
    fn default() -> Self {
        Linear
    }

    fn forward(&self, inputs: Tensor) -> Tensor {
        let output = inputs
            .iter()
            .map(|input| {
                return *input;
            })
            .collect();

        Tensor::new(output, inputs.shape().to_vec())
    }
}
