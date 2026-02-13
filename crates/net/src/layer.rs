use crate::neuron::Neuron;
pub struct Layer {
    neurons: Vec<Neuron>,
    layer_dim: [usize; 2],
}

impl Layer {
    pub fn default() -> Self {
        Layer { neurons: Vec::new(), layer_dim: [0, 0] }
    }

    pub fn add_neuron(&mut self, neuron: Neuron) {
        if neuron.weights.len() == 0 {
            panic!("Neuron must have at least one weight");
        }
        if self.neurons.len() > 0 && neuron.weights.len() != self.neurons[0].weights.len() {
            panic!("All neurons in a layer must have the same number of weights");
        }
        self.neurons.push(neuron);
        self.layer_dim = [self.neurons.len(), self.neurons[0].weights.len()];
    }

    pub fn forward(&self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layer_dim[1] {
            panic!("Input size must match the number of neurons in the layer");
        }
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs.clone()))
            .collect()
    }
}
