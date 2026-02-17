use crate::layer::Layer;
use ndarray::ArrayD;
pub struct Step;

impl Step {
    pub fn new() -> Self {
        Step
    }
}

impl Layer for Step {
    fn forward(&mut self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        inputs.mapv(|input| if input > 0.0 { 1.0 } else { 0.0 })
    }

    fn backward(&mut self, upstream_gradients: ArrayD<f64>) -> ArrayD<f64> {
        ArrayD::zeros(upstream_gradients.raw_dim())
    }
}
