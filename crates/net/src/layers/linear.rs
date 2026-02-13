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
        if neuron.weights.len() == 0 {
            panic!("Neuron must have at least one weight");
        }
        if self.neurons.len() > 0 && neuron.weights.len() != self.neurons[0].weights.len() {
            panic!("All neurons in a layer must have the same number of weights");
        }
        self.neurons.push(neuron);
        self.layer_dim = [self.neurons.len(), self.neurons[0].weights.len()];
    }

    fn forward(&self, inputs: Tensor) -> Tensor {
        if inputs.shape()[0] != self.layer_dim[1] {
            panic!("Input size must match the number of neurons in the layer");
        }
        Tensor::from_vec(
            self.neurons
                .iter()
                .map(|neuron| neuron.forward(inputs.clone()))
                .collect()
        )
    }
}
