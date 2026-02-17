use ndarray::{ Array1, Array2, ArrayD, Axis };

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
        let input_shape = inputs.shape().to_vec();
        if input_shape.is_empty() {
            panic!("Linear forward expects at least a rank-1 tensor");
        }

        let in_features = *input_shape.last().expect("Input shape cannot be empty");
        if in_features != expected_inputs {
            panic!("Input size must match the number of weights in each neuron");
        }

        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>();
        let x = inputs
            .into_shape_with_order((batch_size.max(1), expected_inputs))
            .expect("Linear forward failed to flatten input");

        let mut out = x.dot(&self.weights.t());
        out += &self.biases;

        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(self.layer_dim[0]);
        out.into_shape_with_order(output_shape)
            .expect("Linear forward failed to reshape output")
            .into_dyn()
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        if self.neurons.is_empty() {
            panic!("Layer has no neurons");
        }

        let out_dim = self.layer_dim[0];
        let in_dim = self.layer_dim[1];

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

        if *input_shape.last().expect("Input shape cannot be empty") != in_dim {
            panic!("Input size must match the number of weights in each neuron");
        }

        if *grad_shape.last().expect("Gradient shape cannot be empty") != out_dim {
            panic!("Gradient size must match output dimension");
        }

        let batch_size = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);

        let x = inputs
            .into_shape_with_order((batch_size, in_dim))
            .expect("Linear backward failed to flatten input");
        let dy = gradients
            .into_shape_with_order((batch_size, out_dim))
            .expect("Linear backward failed to flatten gradients");

        let input_grads = dy.dot(&self.weights);
        let weight_grads = dy.t().dot(&x);
        let bias_grads = dy.sum_axis(Axis(0));

        self.weights -= &weight_grads;
        self.biases -= &bias_grads;

        self.sync_neurons();
        input_grads
            .into_shape_with_order(input_shape)
            .expect("Linear backward failed to reshape input gradients")
            .into_dyn()
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
