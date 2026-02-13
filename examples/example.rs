use net::layer::Linear;
use thermite::net::neuron::Neuron;
use thermite::net::layer::Layer;
use thermite::math::tensor::Tensor;
use net::activation::Step;
use net::activation::Activation;
fn main() {
    let input_1 = Tensor::new(vec![0.2, 0.8, -0.5, 1.0], vec![4]);
    let input_2 = Tensor::from_vec(vec![0.5, -0.91, 0.26, -0.5]);
    let neuron_1 = Neuron::new(input_1, 2.0);
    let neuron_2 = Neuron::new(input_2, 3.0);
    let neuron_3 = Neuron::new(Tensor::from_vec(vec![-0.26, -0.27, 0.17, 0.87]), 0.5);
    let mut layer = Linear::default();
    layer.add_neuron(neuron_1);
    layer.add_neuron(neuron_2);
    layer.add_neuron(neuron_3);
    let output = layer.forward(Tensor::from_vec(vec![1.0, 2.0, 3.0, 2.5]));
    println!("Output: {:?}", output);

    let batch = Tensor::from_vec2(
        vec![vec![1.0, 2.0, 3.0, 2.5], vec![2.0, 5.0, -1.0, 2.0], vec![-1.5, 2.7, 3.3, -0.8]]
    );
    let batch_output = layer.forward(batch.clone());
    println!("Batch output: {:?}", batch_output);
    let activation_output = Step::new().forward(batch_output);
    println!("Activation output: {:?}", activation_output);
}
