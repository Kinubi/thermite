use ndarray::{ Array1, Array2, ArrayD, Axis };

use crate::layer::Layer;

pub struct Linear {
    input_dim: usize,
    output_dim: usize,
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl Default for Linear {
    fn default() -> Self {
        Self {
            input_dim: 0,
            output_dim: 0,
            weights: Array2::zeros((0, 0)),
            biases: Array1::zeros(0),
        }
    }
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        if input_dim == 0 || output_dim == 0 {
            panic!("input_dim and output_dim must be greater than zero");
        }

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| rand::random::<f64>());
        let biases = Array1::from_shape_fn(output_dim, |_| rand::random::<f64>());

        Self {
            input_dim,
            output_dim,
            weights,
            biases,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        if self.input_dim == 0 || self.output_dim == 0 {
            panic!("Linear layer is uninitialized. Use Linear::new(input_dim, output_dim)");
        }

        let input_shape = inputs.shape().to_vec();
        if input_shape.is_empty() {
            panic!("Linear forward expects at least a rank-1 tensor");
        }

        let in_features = *input_shape.last().expect("Input shape cannot be empty");
        if in_features != self.input_dim {
            panic!("Input size must match the layer input dimension");
        }

        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>();
        let x = inputs
            .into_shape_with_order((batch_size.max(1), self.input_dim))
            .expect("Linear forward failed to flatten input");

        let mut out = x.dot(&self.weights.t());
        out += &self.biases;

        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(self.output_dim);
        out.into_shape_with_order(output_shape)
            .expect("Linear forward failed to reshape output")
            .into_dyn()
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        if self.input_dim == 0 || self.output_dim == 0 {
            panic!("Linear layer is uninitialized. Use Linear::new(input_dim, output_dim)");
        }

        let input_shape = inputs.shape().to_vec();
        let grad_shape = gradients.shape().to_vec();
        if input_shape.is_empty() || grad_shape.is_empty() {
            panic!("Linear backward expects at least rank-1 tensors");
        }

        if input_shape.len() != grad_shape.len() {
            panic!("Input and gradient ranks must match");
        }

        if input_shape[..input_shape.len() - 1] != grad_shape[..grad_shape.len() - 1] {
            panic!("Gradient batch shape must match input batch shape");
        }

        if *input_shape.last().expect("Input shape cannot be empty") != self.input_dim {
            panic!("Input size must match the layer input dimension");
        }

        if *grad_shape.last().expect("Gradient shape cannot be empty") != self.output_dim {
            panic!("Gradient size must match output dimension");
        }

        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);

        let x = inputs
            .into_shape_with_order((batch_size, self.input_dim))
            .expect("Linear backward failed to flatten input");
        let dy = gradients
            .into_shape_with_order((batch_size, self.output_dim))
            .expect("Linear backward failed to flatten gradients");

        let input_grads = dy.dot(&self.weights);
        let weight_grads = dy.t().dot(&x);
        let bias_grads = dy.sum_axis(Axis(0));

        self.weights -= &weight_grads;
        self.biases -= &bias_grads;
        input_grads
            .into_shape_with_order(input_shape)
            .expect("Linear backward failed to reshape input gradients")
            .into_dyn()
    }
}
