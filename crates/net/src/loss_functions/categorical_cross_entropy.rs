use crate::loss::Loss;
use math::tensor::Tensor;
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

    fn forward(&self, inputs: Tensor, targets: Tensor) -> Tensor {
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

        Tensor::from_vec(vec![loss / (batch_size as f64)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorical_cross_entropy() {
        let loss_fn = CategoricalCrossEntropy::new();
        let inputs = Tensor::from_vec(vec![0.1, 0.5, 0.4]);
        let targets = Tensor::from_vec(vec![0.0, 1.0, 0.0]);
        let loss = loss_fn.forward(inputs, targets);
        let expected = -(0.5_f64).ln();
        assert!((loss.get(vec![0]) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_categorical_cross_entropy_batch() {
        let loss_fn = CategoricalCrossEntropy::new();
        let inputs = Tensor::from_vec2(vec![vec![0.1, 0.5, 0.4], vec![0.8, 0.1, 0.1]]);
        let targets = Tensor::from_vec2(vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]]);

        let loss = loss_fn.forward(inputs, targets);
        let expected = (-(0.5_f64).ln() - (0.8_f64).ln()) / 2.0;
        assert!((loss.get(vec![0]) - expected).abs() < 1e-6);
    }
}
