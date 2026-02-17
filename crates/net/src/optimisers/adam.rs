use crate::optimiser::Optimiser;
use ndarray::ArrayD;

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl Default for Adam {
    fn default() -> Self {
        Adam::new(0.001, 0.9, 0.999, 1e-8)
    }
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }
}

impl Optimiser for Adam {
    fn forward(&self, _inputs: ArrayD<f64>, _targets: ArrayD<f64>) -> ArrayD<f64> {
        unimplemented!("Adam optimiser forward pass is not implemented yet")
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        unimplemented!("Adam optimiser backward pass is not implemented yet")
    }
}
