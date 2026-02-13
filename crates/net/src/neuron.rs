pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: Vec<f64>) -> f64 {
        let sum: f64 = self.weights
            .iter()
            .zip(inputs.iter())
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
        let neuron = Neuron::new(vec![0.5, 0.3], 0.1);
        let output = neuron.forward(vec![1.0, 2.0]);
        assert!((output - 1.2).abs() < 1e-12);
    }
}
