use math::tensor::Tensor;
use crate::activation::Activation;
pub struct Step;

impl Activation for Step {
    fn new() -> Self {
        Step
    }

    fn forward(&self, inputs: Tensor) -> Tensor {
        let output = inputs
            .iter()
            .map(|input| {
                if *input > 0.0 {
                    return 1.0;
                } else {
                    return 0.0;
                }
            })
            .collect();

        Tensor::from_vec(output)
    }
}
