use crate::loss::Loss;
use ndarray::{ arr1, ArrayD };
pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for CategoricalCrossEntropy {
    fn default() -> Self {
        Self
    }

    fn forward(&self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64> {
        if inputs.shape() != targets.shape() {
            panic!("Inputs and targets must have the same shape");
        }

        let shape = inputs.shape();
        let rank = shape.len();
        if rank != 1 && rank != 2 {
            panic!(
                "CategoricalCrossEntropy only supports rank-1 or rank-2 tensors, got shape {:?}",
                shape
            );
        }

        let batch_size = if rank == 2 { shape[0] } else { 1 };
        let epsilon = 1e-12;
        let mut loss = 0.0;

        for (target, input) in targets.iter().zip(inputs.iter()) {
            let p = (*input).clamp(epsilon, 1.0 - epsilon);
            loss -= *target * p.ln();
        }

        arr1(&[loss / (batch_size as f64)]).into_dyn()
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        unimplemented!("Backward pass for CategoricalCrossEntropy is not implemented yet")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ arr1, arr2 };

    #[test]
    fn test_categorical_cross_entropy() {
        let loss_fn = CategoricalCrossEntropy::new();
        let inputs = arr1(&[0.1, 0.5, 0.4]).into_dyn();
        let targets = arr1(&[0.0, 1.0, 0.0]).into_dyn();
        let loss = loss_fn.forward(inputs, targets);
        let expected = -(0.5_f64).ln();
        assert!((loss[[0]] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_categorical_cross_entropy_batch() {
        let loss_fn = CategoricalCrossEntropy::new();
        let inputs = arr2(
            &[
                [0.1, 0.5, 0.4],
                [0.8, 0.1, 0.1],
            ]
        ).into_dyn();
        let targets = arr2(
            &[
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ).into_dyn();

        let loss = loss_fn.forward(inputs, targets);
        let expected = (-(0.5_f64).ln() - (0.8_f64).ln()) / 2.0;
        assert!((loss[[0]] - expected).abs() < 1e-12);
    }
}
