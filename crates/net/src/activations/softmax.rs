use math::tensor::Tensor;
use crate::activation::Activation;
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

    fn forward(&self, inputs: Tensor) -> Tensor {
        match inputs.shape_slice() {
            [_n] => {
                let max = inputs
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = inputs.iter().map(|&x| (x - max).exp()).sum();
                let out: Vec<f64> = inputs.iter().map(|&x| (x - max).exp() / sum).collect();
                Tensor::from_vec(out)
            }
            [rows, _cols] => {
                let mut output: Vec<Vec<f64>> = Vec::with_capacity(*rows);
                for r in 0..*rows {
                    let row_view = inputs.narrow(0, r, 1); // shape [1, cols]
                    let max = row_view
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = row_view.iter().map(|&x| (x - max).exp()).sum();
                    let row_out: Vec<f64> = row_view
                        .iter()
                        .map(|&x| (x - max).exp() / sum)
                        .collect();
                    output.push(row_out);
                }
                Tensor::from_vec2(output)
            }
            other => panic!("Softmax expects a 1D or 2D tensor, got shape {other:?}"),
        }
    }
}
