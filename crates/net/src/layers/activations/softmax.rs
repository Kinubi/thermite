use crate::layer::Layer;
use ndarray::ArrayD;
pub struct Softmax {
    cached_outputs: Option<ArrayD<f64>>,
}

impl Softmax {
    pub fn new() -> Self {
        Softmax { cached_outputs: None }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Softmax {
    fn forward(&mut self, inputs: ArrayD<f64>) -> ArrayD<f64> {
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

        let output = out
            .into_shape_with_order(input_shape)
            .expect("Softmax failed to reshape output")
            .into_dyn();
        self.cached_outputs = Some(output.clone());
        output
    }

    fn backward(&mut self, upstream_gradients: ArrayD<f64>) -> ArrayD<f64> {
        let s_cached = self.cached_outputs.take().expect("Softmax backward called before forward");
        let output_shape = s_cached.shape().to_vec();
        let grad_shape = upstream_gradients.shape().to_vec();

        if output_shape.is_empty() || grad_shape.is_empty() {
            panic!("Softmax backward expects at least rank-1 tensors");
        }

        if output_shape != grad_shape {
            panic!("Softmax backward expects inputs and upstream gradients to have the same shape");
        }

        let classes = *output_shape.last().expect("Output shape cannot be empty");
        let batch_size = output_shape[..output_shape.len() - 1].iter().product::<usize>().max(1);

        let s = s_cached
            .into_shape_with_order((batch_size, classes))
            .expect("Softmax backward failed to flatten outputs");
        let g = upstream_gradients
            .into_shape_with_order((batch_size, classes))
            .expect("Softmax backward failed to flatten gradients");

        let mut d_inputs = s.clone();
        for row in 0..batch_size {
            let s_row = s.row(row);
            let g_row = g.row(row);
            let dot = s_row.dot(&g_row);
            for col in 0..classes {
                d_inputs[(row, col)] = s_row[col] * (g_row[col] - dot);
            }
        }

        d_inputs
            .into_shape_with_order(output_shape)
            .expect("Softmax backward failed to reshape output gradients")
            .into_dyn()
    }
}
