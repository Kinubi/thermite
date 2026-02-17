use ndarray::ArrayD;

pub use crate::layers::Linear;

pub trait Layer {
    fn forward(&mut self, inputs: ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&mut self, gradients: ArrayD<f64>) -> ArrayD<f64>;

    fn zero_grad(&mut self) {}

    fn step(&mut self, _learning_rate: f64) {}
}
