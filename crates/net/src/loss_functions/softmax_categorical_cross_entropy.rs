use crate::loss::Loss;
use ndarray::{ arr1, Array2, ArrayD };

pub struct SoftmaxCategoricalCrossEntropy;

impl Default for SoftmaxCategoricalCrossEntropy {
    fn default() -> Self {
        Self
    }
}

impl SoftmaxCategoricalCrossEntropy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Loss for SoftmaxCategoricalCrossEntropy {
    fn forward(&self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64> {
        let input_shape = inputs.shape().to_vec();
        if input_shape.is_empty() {
            panic!("SoftmaxCategoricalCrossEntropy expects at least a rank-1 tensor");
        }

        let probs = softmax_last_axis(inputs);
        let targets = targets_to_one_hot(targets, &input_shape);
        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);
        let epsilon = 1e-12;

        let mut loss = 0.0;
        for (target, prob) in targets.iter().zip(probs.iter()) {
            let p = (*prob).clamp(epsilon, 1.0 - epsilon);
            loss -= *target * p.ln();
        }

        arr1(&[loss / (batch_size as f64)]).into_dyn()
    }

    fn backward(&mut self, inputs: ArrayD<f64>, targets: ArrayD<f64>) -> ArrayD<f64> {
        let input_shape = inputs.shape().to_vec();
        if input_shape.is_empty() {
            panic!("SoftmaxCategoricalCrossEntropy backward expects at least a rank-1 tensor");
        }

        let probs = softmax_last_axis(inputs);
        let targets = targets_to_one_hot(targets, &input_shape);
        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);

        (probs - targets) / (batch_size as f64)
    }
}

fn softmax_last_axis(inputs: ArrayD<f64>) -> ArrayD<f64> {
    let input_shape = inputs.shape().to_vec();
    if input_shape.is_empty() {
        panic!("Softmax expects at least a rank-1 tensor");
    }

    let classes = *input_shape.last().expect("Input shape cannot be empty");
    let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);

    let mut out = inputs
        .into_shape_with_order((batch_size, classes))
        .expect("Softmax failed to flatten input");

    for mut row in out.rows_mut() {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for value in &mut row {
            *value = (*value - max).exp();
            sum += *value;
        }
        for value in &mut row {
            *value /= sum;
        }
    }

    out.into_shape_with_order(input_shape).expect("Softmax failed to reshape output").into_dyn()
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
    fn test_softmax_cce_forward_with_one_hot_targets() {
        let loss_fn = SoftmaxCategoricalCrossEntropy::new();
        let logits = arr2(
            &[
                [1.0, 2.0, 0.5],
                [3.0, 0.5, -1.0],
            ]
        ).into_dyn();
        let targets = arr2(
            &[
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ).into_dyn();

        let loss = loss_fn.forward(logits, targets);
        assert!(loss[[0]].is_finite());
        assert!(loss[[0]] > 0.0);
    }

    #[test]
    fn test_softmax_cce_backward_matches_softmax_minus_one_hot() {
        let mut loss_fn = SoftmaxCategoricalCrossEntropy::new();
        let logits = arr2(
            &[
                [1.0, 2.0, 0.5],
                [3.0, 0.5, -1.0],
            ]
        ).into_dyn();
        let targets = arr1(&[1.0, 0.0]).into_dyn();

        let grad = loss_fn.backward(logits.clone(), targets);
        let probs = softmax_last_axis(logits)
            .into_shape_with_order((2, 3))
            .expect("Expected 2x3 probabilities");

        let expected =
            (probs -
                arr2(
                    &[
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                )) /
            2.0;

        let grad = grad.into_shape_with_order((2, 3)).expect("Expected 2x3 gradients");

        for (a, b) in grad.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
