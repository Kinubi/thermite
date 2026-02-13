use math::tensor::Tensor;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(weights: Tensor, bias: f64) -> Self {
        Neuron { weights: weights.data().clone(), bias }
    }

    pub fn forward(&self, inputs: Tensor) -> f64 {
        let sum: f64 = self.weights
            .iter()
            .zip(inputs.data().iter())
            .map(|(w, i)| w * i)
            .sum();
        sum + self.bias
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
