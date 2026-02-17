use crate::activation::Activation;
use ndarray::ArrayD;
pub struct Step;

impl Step {
    pub fn new() -> Self {
        Step
    }
}

impl Activation for Step {
    fn default() -> Self {
        Step
    }

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        inputs.mapv(|input| if input > 0.0 { 1.0 } else { 0.0 })
    }

    fn backward(&mut self, inputs: ArrayD<f64>, upstream_gradients: ArrayD<f64>) -> ArrayD<f64> {
        if inputs.shape() != upstream_gradients.shape() {
            panic!("Step backward expects inputs and upstream gradients to have the same shape");
        }

        ArrayD::zeros(inputs.raw_dim())
    }
}
