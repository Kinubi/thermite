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

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        let sigmoid_outputs = self.forward(inputs);
        sigmoid_outputs.mapv(|output| output * (1.0 - output)) * &gradients
    }
}
