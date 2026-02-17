use ndarray::ArrayD;

pub use crate::loss_functions::categorical_cross_entropy::CategoricalCrossEntropy;
pub use crate::loss_functions::softmax_categorical_cross_entropy::SoftmaxCategoricalCrossEntropy;

pub trait Loss {
    fn forward(&mut self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&mut self) -> ArrayD<f64>;
}
