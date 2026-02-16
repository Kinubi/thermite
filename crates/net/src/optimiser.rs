use ndarray::ArrayD;

pub use crate::optimisers::Adam;

pub trait Optimiser {
    fn default() -> Self;
    fn forward(&self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64>;
}
