use crate::loss::Loss;
use ndarray::{ arr1, Array2, ArrayD };
pub struct CategoricalCrossEntropy {
    cached_inputs: Option<ArrayD<f64>>,
    cached_targets: Option<ArrayD<f64>>,
}

impl Default for CategoricalCrossEntropy {
    fn default() -> Self {
        Self {
            cached_inputs: None,
            cached_targets: None,
        }
    }
}

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Loss for CategoricalCrossEntropy {
    fn forward(&mut self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64> {
        let input_shape = inputs.shape().to_vec();
        if input_shape.is_empty() {
            panic!("CategoricalCrossEntropy expects at least a rank-1 tensor");
        }

        let targets = targets_to_one_hot(targets, &input_shape);

        self.cached_inputs = Some(inputs.clone());
        self.cached_targets = Some(targets.clone());

        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);
        let epsilon = 1e-12;
        let mut loss = 0.0;

        for (target, input) in targets.iter().zip(inputs.iter()) {
            let p = (*input).clamp(epsilon, 1.0 - epsilon);
            loss -= *target * p.ln();
        }

        arr1(&[loss / (batch_size as f64)]).into_dyn()
    }

    fn backward(&mut self) -> ArrayD<f64> {
        let inputs = self.cached_inputs.take().expect("Loss backward called before forward");
        let targets = self.cached_targets.take().expect("Loss backward called before forward");

        let input_shape = inputs.shape().to_vec();
        if input_shape.is_empty() {
            panic!("CategoricalCrossEntropy backward expects at least a rank-1 tensor");
        }

        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);
        let epsilon = 1e-12;

        let safe_inputs = inputs.mapv(|p| p.clamp(epsilon, 1.0 - epsilon));
        -(targets / safe_inputs) / (batch_size as f64)
    }
}

fn targets_to_one_hot(targets: ArrayD<f64>, prob_shape: &[usize]) -> ArrayD<f64> {
    if prob_shape.is_empty() {
        panic!("Probability shape must be at least rank-1");
    }

    if targets.shape() == prob_shape {
        return targets;
    }

    let num_classes = *prob_shape.last().expect("Probability shape cannot be empty");
    let batch_shape = &prob_shape[..prob_shape.len() - 1];

    let is_single_sample_index = prob_shape.len() == 1 && targets.ndim() == 1 && targets.len() == 1;
    if !is_single_sample_index && targets.shape() != batch_shape {
        panic!(
            "Targets shape must match probabilities shape {:?} (one-hot) or batch shape {:?} (class indices), got {:?}",
            prob_shape,
            batch_shape,
            targets.shape()
        );
    }

    let batch_size = batch_shape.iter().product::<usize>().max(1);
    let flat_targets = targets
        .into_shape_with_order(batch_size)
        .expect("Failed to flatten target indices");

    let mut one_hot = Array2::<f64>::zeros((batch_size, num_classes));
    for (row, &class_id_f64) in flat_targets.iter().enumerate() {
        if !class_id_f64.is_finite() {
            panic!("Target index must be finite");
        }
        let class_id = class_id_f64 as usize;
        if ((class_id as f64) - class_id_f64).abs() > f64::EPSILON {
            panic!("Target index must be an integer");
        }
        if class_id >= num_classes {
            panic!("Target class index out of range");
        }
        one_hot[(row, class_id)] = 1.0;
    }

    let mut output_shape = batch_shape.to_vec();
    output_shape.push(num_classes);
    one_hot
        .into_shape_with_order(output_shape)
        .expect("Failed to reshape one-hot targets")
        .into_dyn()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ arr1, arr2 };

    #[test]
    fn test_categorical_cross_entropy() {
        let mut loss_fn = CategoricalCrossEntropy::new();
        let inputs = arr1(&[0.1, 0.5, 0.4]).into_dyn();
        let targets = arr1(&[0.0, 1.0, 0.0]).into_dyn();
        let loss = loss_fn.forward(inputs, targets);
        let expected = -(0.5_f64).ln();
        assert!((loss[[0]] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_categorical_cross_entropy_batch() {
        let mut loss_fn = CategoricalCrossEntropy::new();
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

    #[test]
    fn test_categorical_cross_entropy_batch_with_class_indices() {
        let mut loss_fn = CategoricalCrossEntropy::new();
        let inputs = arr2(
            &[
                [0.1, 0.5, 0.4],
                [0.8, 0.1, 0.1],
            ]
        ).into_dyn();
        let targets = arr1(&[1.0, 0.0]).into_dyn();

        let loss = loss_fn.forward(inputs, targets);
        let expected = (-(0.5_f64).ln() - (0.8_f64).ln()) / 2.0;
        assert!((loss[[0]] - expected).abs() < 1e-12);
    }
}
