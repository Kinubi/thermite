use ndarray::Array1;

pub struct Neuron {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(weights: Array1<f64>, bias: f64) -> Self {
        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: Array1<f64>) -> f64 {
        self.weights.dot(&inputs) + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_neuron_forward() {
        let neuron = Neuron::new(arr1(&[0.5, 0.3]), 0.1);
        let output = neuron.forward(arr1(&[1.0, 2.0]));
        assert!((output - 1.2).abs() < 1e-12);
    }
}
