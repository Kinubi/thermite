use math::tensor::Tensor;

use crate::layer::Layer;
use crate::neuron::Neuron;

pub struct Linear {
    neurons: Vec<Neuron>,
    layer_dim: [usize; 2],
    weights: Tensor,
    biases: Vec<f64>,
}

impl Default for Linear {
    fn default() -> Self {
        Self {
            neurons: Vec::new(),
            layer_dim: [0, 0],
            weights: Tensor::zeros(vec![0, 0]),
            biases: Vec::new(),
        }
    }
}

impl Layer for Linear {
    fn new(input_len: usize, num_neurons: usize) -> Self {
        let mut layer = Linear::default();
        layer.weights = Tensor::zeros(vec![0, input_len]);
        for _ in 0..num_neurons {
            let neuron = Neuron::new(Tensor::zeros(vec![input_len]), 0.0);
            layer.add_neuron(neuron);
        }
        layer
    }

    fn add_neuron(&mut self, neuron: Neuron) {
        if neuron.weights.shape()[0] == 0 {
            panic!("Neuron must have at least one weight");
        }
        if
            self.neurons.len() > 0 &&
            neuron.weights.shape()[0] != self.neurons[0].weights.shape()[0]
        {
            panic!("All neurons in a layer must have the same number of weights");
        }
        self.neurons.push(neuron);
        self.layer_dim = [self.neurons.len(), self.neurons[0].weights.shape()[0]];
        self.weights.push(self.neurons.last().unwrap().weights.clone());
        self.biases.push(self.neurons.last().unwrap().bias);
    }

    fn forward(&self, inputs: Tensor) -> Tensor {
        if self.neurons.is_empty() {
            panic!("Layer has no neurons");
        }

        let expected_inputs = self.layer_dim[1];
        match inputs.shape().as_slice() {
            [n] => {
                if *n != expected_inputs {
                    panic!("Input size must match the number of weights in each neuron");
                }
            }
            [_batch, n] => {
                if *n != expected_inputs {
                    panic!("Input size must match the number of weights in each neuron");
                }
            }
            _ => panic!("Linear forward expects a 1D or 2D tensor"),
        }

        let bias = Tensor::from_vec(self.biases.clone());

        let mut out = match inputs.shape().as_slice() {
            [n] => {
                if *n != expected_inputs {
                    panic!("Input size must match the number of weights in each neuron");
                }
                self.weights.matmul(&inputs)
            }
            [_batch, n] => {
                if *n != expected_inputs {
                    panic!("Input size must match the number of weights in each neuron");
                }
                inputs.matmul(&self.weights.transpose())
            }
            _ => panic!("Linear forward expects a 1D or 2D tensor"),
        };

        let out_shape = out.shape().clone();
        match out_shape.as_slice() {
            [out_dim] => {
                if *out_dim != bias.shape()[0] {
                    panic!("Bias dimension must match output dimension");
                }
                out += &bias;
            }
            [batch, out_dim] => {
                if *out_dim != bias.shape()[0] {
                    panic!("Bias dimension must match output dimension");
                }
                let out_dim = *out_dim;
                for row in out.data_mut().chunks_mut(out_dim) {
                    for (j, v) in row.iter_mut().enumerate() {
                        *v += bias.data_slice()[j];
                    }
                }
                let _ = batch; // keep variable used for clarity
            }
            _ => unreachable!(),
        }

        out
    }
}
