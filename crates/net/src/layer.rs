use ndarray::ArrayD;

use crate::neuron::Neuron;

pub use crate::layers::Linear;

pub trait Layer {
    fn new(input_len: usize, num_neurons: usize) -> Self;
    fn add_neuron(&mut self, neuron: Neuron);
    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64>;
}
