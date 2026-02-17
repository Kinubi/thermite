use crate::layer::Layer;
use ndarray::ArrayD;

pub struct ReLU {
    cached_inputs: Option<ArrayD<f64>>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { cached_inputs: None }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ReLU {
    fn forward(&mut self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        self.cached_inputs = Some(inputs.clone());
        inputs.mapv(|input| input.max(0.0))
    }

    fn backward(&mut self, upstream_gradients: ArrayD<f64>) -> ArrayD<f64> {
        let inputs = self.cached_inputs.take().expect("ReLU backward called before forward");

        if inputs.shape() != upstream_gradients.shape() {
            panic!("ReLU backward expects inputs and upstream gradients to have the same shape");
        }

        let local_grads = inputs.mapv(|input| if input > 0.0 { 1.0 } else { 0.0 });
        local_grads * upstream_gradients
    }
}
