use crate::activation::Activation;
use ndarray::ArrayD;
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

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        inputs.mapv(|input| 1.0 / (1.0 + (-input).exp()))
    }
}
