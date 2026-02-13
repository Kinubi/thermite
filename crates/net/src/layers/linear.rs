use math::tensor::Tensor;

use crate::layer::Layer;
use crate::neuron::Neuron;

pub struct Linear {
    neurons: Vec<Neuron>,
    layer_dim: [usize; 2],
}

impl Default for Linear {
    fn default() -> Self {
        Self {
            neurons: Vec::new(),
            layer_dim: [0, 0],
        }
    }
}

impl Layer for Linear {
    fn new(input_len: usize, num_neurons: usize) -> Self {
        let mut layer = Linear::default();
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

        if inputs.shape().len() == 2 {
            let cols = inputs.shape()[1];
            let outputs: Vec<Vec<f64>> = inputs
                .data()
                .chunks(cols)
                .map(|row| {
                    let row_tensor = Tensor::from_vec(row.to_vec());
                    self.neurons
                        .iter()
                        .map(|neuron| neuron.forward(row_tensor.clone()))
                        .collect()
                })
                .collect();

            return Tensor::from_vec2(outputs);
        }

        Tensor::from_vec(
            self.neurons
                .iter()
                .map(|neuron| neuron.forward(inputs.clone()))
                .collect()
        )
    }
}
