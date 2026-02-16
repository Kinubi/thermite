use ndarray::ArrayD;

pub use crate::loss_functions::categorical_cross_entropy::CategoricalCrossEntropy;

pub trait Loss {
    fn default() -> Self;
    fn forward(&self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64>;
}
