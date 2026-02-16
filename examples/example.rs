use net::layer::Linear;
use thermite::net::neuron::Neuron;
use thermite::net::layer::Layer;
use net::activation::Softmax;
use net::activation::Activation;
use net::loss::CategoricalCrossEntropy;
use net::loss::Loss;
use thermite::math::ndarray::{ arr1, arr2 };

fn main() {
    let input_1 = arr1(&[0.2, 0.8, -0.5, 1.0]);
    let input_2 = arr1(&[0.5, -0.91, 0.26, -0.5]);
    let neuron_1 = Neuron::new(input_1, 2.0);
    let neuron_2 = Neuron::new(input_2, 3.0);
    let neuron_3 = Neuron::new(arr1(&[-0.26, -0.27, 0.17, 0.87]), 0.5);
    let mut layer = Linear::default();
    layer.add_neuron(neuron_1);
    layer.add_neuron(neuron_2);
    layer.add_neuron(neuron_3);
    let output = layer.forward(arr1(&[1.0, 2.0, 3.0, 2.5]).into_dyn());
    println!("Output: {:?}", output);

    let batch = arr2(
        &[
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
        ]
    ).into_dyn();
    let batch_output = layer.forward(batch.clone());
    println!("Batch output: {:?}", batch_output);
    let activation_output = Softmax::new().forward(batch_output);
    println!("Activation output: {:?}", activation_output);
    let loss = CategoricalCrossEntropy::new().forward(
        activation_output,
        arr2(
            &[
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ).into_dyn()
    );
    println!("Loss: {:?}", loss);
}
