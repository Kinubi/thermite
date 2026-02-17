use crate::activation::Activation;
use ndarray::ArrayD;
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Softmax
    }
}

impl Activation for Softmax {
    fn default() -> Self {
        Softmax
    }

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
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
            for v in &mut row {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in &mut row {
                *v /= sum;
            }
        }

        out.into_shape_with_order(input_shape)
            .expect("Softmax failed to reshape output")
            .into_dyn()
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        unimplemented!("Backward pass for Softmax is not implemented yet")
    }
}
