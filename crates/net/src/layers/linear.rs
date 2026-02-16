use ndarray::{ Array1, Array2, ArrayD, Axis, Ix1, Ix2 };

use crate::layer::Layer;
use crate::neuron::Neuron;

pub struct Linear {
    neurons: Vec<Neuron>,
    layer_dim: [usize; 2],
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl Default for Linear {
    fn default() -> Self {
        Self {
            neurons: Vec::new(),
            layer_dim: [0, 0],
            weights: Array2::zeros((0, 0)),
            biases: Array1::zeros(0),
        }
    }
}

impl Layer for Linear {
    fn new(input_len: usize, num_neurons: usize) -> Self {
        let mut layer = Linear::default();

        layer.weights = Array2::zeros((0, input_len));
        for _ in 0..num_neurons {
            let v: Vec<f64> = (0..input_len).map(|_| rand::random::<f64>()).collect();
            let b = rand::random::<f64>();
            let neuron = Neuron::new(Array1::from_vec(v), b);
            layer.add_neuron(neuron);
        }
        layer
    }

    fn add_neuron(&mut self, neuron: Neuron) {
        if neuron.weights.len() == 0 {
            panic!("Neuron must have at least one weight");
        }
        if self.neurons.len() > 0 && neuron.weights.len() != self.neurons[0].weights.len() {
            panic!("All neurons in a layer must have the same number of weights");
        }
        self.neurons.push(neuron);
        self.layer_dim = [self.neurons.len(), self.neurons[0].weights.len()];

        self.weights = Array2::zeros((self.layer_dim[0], self.layer_dim[1]));
        self.biases = Array1::zeros(self.layer_dim[0]);
        for (i, n) in self.neurons.iter().enumerate() {
            self.weights.row_mut(i).assign(&n.weights.view());
            self.biases[i] = n.bias;
        }
    }

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        if self.neurons.is_empty() {
            panic!("Layer has no neurons");
        }

        let expected_inputs = self.layer_dim[1];
        match inputs.ndim() {
            1 => {
                let x = inputs
                    .into_dimensionality::<Ix1>()
                    .expect("Linear forward expects rank-1 input");
                if x.len() != expected_inputs {
                    panic!("Input size must match the number of weights in each neuron");
                }
                (&self.weights.dot(&x) + &self.biases).into_dyn()
            }
            2 => {
                let x = inputs
                    .into_dimensionality::<Ix2>()
                    .expect("Linear forward expects rank-2 input");
                if x.shape()[1] != expected_inputs {
                    panic!("Input size must match the number of weights in each neuron");
                }
                let mut out = x.dot(&self.weights.t());
                for mut row in out.rows_mut() {
                    row += &self.biases;
                }
                out.into_dyn()
            }
            _ => panic!("Linear forward expects a 1D or 2D tensor"),
        }
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        if self.neurons.is_empty() {
            panic!("Layer has no neurons");
        }

        let out_dim = self.layer_dim[0];
        let in_dim = self.layer_dim[1];

        // Validate shapes for x and dY.
        let input_rank = inputs.ndim();
        let grad_rank = gradients.ndim();

        if input_rank != grad_rank {
            panic!("Linear backward expects x and gradients to be both 1D or both 2D");
        }

        match input_rank {
            1 => {
                let x = inputs.view().into_dimensionality::<Ix1>().expect("Expected 1D input");
                let dy = gradients
                    .view()
                    .into_dimensionality::<Ix1>()
                    .expect("Expected 1D gradients");

                if x.len() != in_dim {
                    panic!("Input size must match the number of weights in each neuron");
                }
                if dy.len() != out_dim {
                    panic!("Gradient size must match output dimension");
                }

                let input_grads = dy.dot(&self.weights).into_dyn();
                let weight_grads = dy.insert_axis(Axis(1)).dot(&x.insert_axis(Axis(0)));
                let bias_grads = dy.to_owned();

                self.weights -= &weight_grads;
                self.biases -= &bias_grads;

                self.sync_neurons();
                input_grads
            }
            2 => {
                let x = inputs.view().into_dimensionality::<Ix2>().expect("Expected 2D input");
                let dy = gradients
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("Expected 2D gradients");

                if x.shape()[1] != in_dim {
                    panic!("Input size must match the number of weights in each neuron");
                }
                if dy.shape()[1] != out_dim {
                    panic!("Gradient size must match output dimension");
                }
                if x.shape()[0] != dy.shape()[0] {
                    panic!("Gradient batch size must match input batch size");
                }

                let input_grads = dy.dot(&self.weights).into_dyn();
                let weight_grads = dy.t().dot(&x);
                let bias_grads = dy.sum_axis(Axis(0));

                self.weights -= &weight_grads;
                self.biases -= &bias_grads;

                self.sync_neurons();
                input_grads
            }
            _ => panic!("Linear backward expects x and gradients to be both 1D or both 2D"),
        }
    }
}

impl Linear {
    fn sync_neurons(&mut self) {
        if self.neurons.len() != self.layer_dim[0] {
            panic!("Internal error: neuron count does not match layer_dim[0]");
        }

        for i in 0..self.layer_dim[0] {
            self.neurons[i].weights = self.weights.row(i).to_owned();
            self.neurons[i].bias = self.biases[i];
        }
    }
}
