use crate::activation::Activation;
use ndarray::{ Array1, ArrayD, Ix1, Ix2 };
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
        match inputs.ndim() {
            1 => {
                let x = inputs.into_dimensionality::<Ix1>().expect("Softmax expects rank-1 input");
                softmax_1d(&x).into_dyn()
            }
            2 => {
                let x = inputs.into_dimensionality::<Ix2>().expect("Softmax expects rank-2 input");
                let mut out = x.clone();
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
                out.into_dyn()
            }
            _ => panic!("Softmax expects a 1D or 2D tensor"),
        }
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        unimplemented!("Backward pass for Softmax is not implemented yet")
    }
}

fn softmax_1d(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps = x.mapv(|v| (v - max).exp());
    let sum = exps.sum();
    exps.mapv(|v| v / sum)
}
