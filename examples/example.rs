use thermite::net::neuron::Neuron;
use thermite::net::layer::Layer;
fn main() {
    let neuron_1 = Neuron::new(vec![0.2, 0.8, -0.5, 1.0], 2.0);
    let neuron_2 = Neuron::new(vec![0.5, -0.91, 0.26, -0.5], 3.0);
    let neuron_3 = Neuron::new(vec![-0.26, -0.27, 0.17, 0.87], 0.5);
    let mut layer = Layer::default();
    layer.add_neuron(neuron_1);
    layer.add_neuron(neuron_2);
    layer.add_neuron(neuron_3);
    let output = layer.forward(vec![1.0, 2.0, 3.0, 2.5]);
    println!("Output: {:?}", output);
}
