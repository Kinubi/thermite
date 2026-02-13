use math::tensor::Tensor;

pub struct Neuron {
    pub weights: Tensor,
    pub bias: f64,
}

impl Neuron {
    pub fn new(weights: Tensor, bias: f64) -> Self {
        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: Tensor) -> f64 {
        self.weights.dot(&inputs) + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_neuron_forward() {
        let neuron = Neuron::new(Tensor::new(vec![0.5, 0.3], vec![2]), 0.1);
        let output = neuron.forward(Tensor::new(vec![1.0, 2.0], vec![2]));
        assert!((output - 1.2).abs() < 1e-12);
    }
}
