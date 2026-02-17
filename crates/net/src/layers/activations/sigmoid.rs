use crate::layer::Layer;
use ndarray::ArrayD;
pub struct Sigmoid {
    cached_outputs: Option<ArrayD<f64>>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid { cached_outputs: None }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        let outputs = inputs.mapv(|input| 1.0 / (1.0 + (-input).exp()));
        self.cached_outputs = Some(outputs.clone());
        outputs
    }

    fn backward(&mut self, upstream_gradients: ArrayD<f64>) -> ArrayD<f64> {
        let sigmoid_outputs = self.cached_outputs
            .take()
            .expect("Sigmoid backward called before forward");

        if sigmoid_outputs.shape() != upstream_gradients.shape() {
            panic!("Sigmoid backward expects inputs and upstream gradients to have the same shape");
        }

        sigmoid_outputs.mapv(|output| output * (1.0 - output)) * upstream_gradients
    }
}
