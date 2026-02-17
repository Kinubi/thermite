use thermite::net::layer::Layer;
use thermite::net::layers::{ Linear, Softmax };
use thermite::net::loss::{ CategoricalCrossEntropy, Loss };
use thermite::net::model::Sequential;
use thermite::math::ndarray::{ arr1, arr2 };

fn main() {
    let mut model = Sequential::new();
    model.add_module(Linear::new(4, 3));
    model.add_module(Softmax::new());

    println!("Training mode: {}", model.is_training());
    model.eval();
    println!("Training mode after eval(): {}", model.is_training());
    model.train();
    println!("Training mode after train(): {}", model.is_training());

    let layer = Linear::new(4, 3);
    let output = layer.forward(arr1(&[1.0, 2.0, 3.0, 2.5]).into_dyn());
    println!("Output: {:?}", output);

    let batch = arr2(
        &[
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8],
        ]
    ).into_dyn();
    let activation_output = model.forward(batch.clone());
    println!("Sequential output: {:?}", activation_output);
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
