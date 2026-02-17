use crate::layer::Layer;
use ndarray::ArrayD;
pub struct Linear;

impl Linear {
    pub fn new() -> Self {
        Linear
    }
}

impl Layer for Linear {
    fn forward(&mut self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        inputs
    }

    fn backward(&mut self, upstream_gradients: ArrayD<f64>) -> ArrayD<f64> {
        upstream_gradients
    }
}
