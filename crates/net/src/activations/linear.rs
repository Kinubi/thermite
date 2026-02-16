use crate::activation::Activation;
use ndarray::ArrayD;
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

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        inputs
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        gradients
    }
}
