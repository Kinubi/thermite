use ndarray::ArrayD;

pub use crate::layers::Linear;

pub trait Layer {
    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64>;
}
