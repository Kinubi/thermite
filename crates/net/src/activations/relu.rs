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

    fn backward(&mut self, inputs: ArrayD<f64>, upstream_gradients: ArrayD<f64>) -> ArrayD<f64> {
        if inputs.shape() != upstream_gradients.shape() {
            panic!("ReLU backward expects inputs and upstream gradients to have the same shape");
        }

        let local_grads = inputs.mapv(|input| if input > 0.0 { 1.0 } else { 0.0 });
        local_grads * upstream_gradients
    }
}
