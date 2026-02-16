use crate::activation::Activation;
use ndarray::ArrayD;

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

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        inputs.mapv(|input| input.max(0.0))
    }
}
