pub mod categorical_cross_entropy;
pub mod softmax_categorical_cross_entropy;

pub use categorical_cross_entropy::CategoricalCrossEntropy;
pub use softmax_categorical_cross_entropy::SoftmaxCategoricalCrossEntropy;

pub mod mse;

pub use mse::MSE;
