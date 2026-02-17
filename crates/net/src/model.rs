pub use crate::module::Module;
use crate::layer::Layer;
use ndarray::ArrayD;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: bool,
}

impl Default for Sequential {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            training: true,
        }
    }
}

impl Sequential {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_layer(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }

    pub fn add_module<M: Module + 'static>(&mut self, module: M) {
        self.layers.push(Box::new(module));
    }

    pub fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        let mut output = inputs;
        for layer in &self.layers {
            output = layer.forward(output);
        }
        output
    }

    pub fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        let mut grad = gradients;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(inputs.clone(), grad);
        }
        grad
    }

    pub fn train(&mut self) {
        self.training = true;
        for layer in self.layers.iter_mut() {
            layer.train();
        }
    }

    pub fn eval(&mut self) {
        self.training = false;
        for layer in self.layers.iter_mut() {
            layer.eval();
        }
    }

    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Layer for Sequential {
    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        Sequential::forward(self, inputs)
    }

    fn backward(&mut self, inputs: ArrayD<f64>, gradients: ArrayD<f64>) -> ArrayD<f64> {
        Sequential::backward(self, inputs, gradients)
    }
}

pub type Model = Sequential;
