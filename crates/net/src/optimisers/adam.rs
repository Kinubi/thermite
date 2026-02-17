use crate::optimiser::Optimiser;
use crate::module::Module;

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl Default for Adam {
    fn default() -> Self {
        Adam::new(0.001, 0.9, 0.999, 1e-8)
    }
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }

    pub fn from_model<M: Module + ?Sized>(model: &M) -> Self {
        <Self as Optimiser>::from_model(model)
    }
}

impl Optimiser for Adam {
    fn from_model<M: Module + ?Sized>(_model: &M) -> Self {
        Self::default()
    }

    fn zero_grad<M: Module + ?Sized>(&mut self, model: &mut M) {
        model.zero_grad();
    }

    fn step<M: Module + ?Sized>(&mut self, _model: &mut M) {
        _model.step(self.learning_rate);
    }
}
