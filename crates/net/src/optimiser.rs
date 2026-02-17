use ndarray::ArrayD;

pub use crate::optimisers::Adam;

pub trait Optimiser {
    fn forward(&self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64>;
}
