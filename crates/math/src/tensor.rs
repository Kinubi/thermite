#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn from_vec(vec: Vec<f64>) -> Self {
        Self { data: vec.clone(), shape: vec![vec.len()] }
    }

    pub fn from_vec2(vec: Vec<Vec<f64>>) -> Self {
        let mut data = Vec::new();
        for pair in vec.clone() {
            data.extend(pair.into_iter());
        }
        Self { data, shape: vec![vec.len(), 2] }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self { data: vec![0.0; size], shape }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self { data: vec![1.0; size], shape }
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            panic!("New shape must have the same number of elements as the original shape");
        }
        self.shape = new_shape;
    }

    pub fn get(&self, indices: Vec<usize>) -> f64 {
        if indices.len() != self.shape.len() {
            panic!("Number of indices must match the number of dimensions");
        }
        let mut flat_index = 0;
        let mut stride = 1;
        for (i, &index) in indices.iter().rev().enumerate() {
            if index >= self.shape[self.shape.len() - 1 - i] {
                panic!("Index out of bounds");
            }
            flat_index += index * stride;
            stride *= self.shape[self.shape.len() - 1 - i];
        }
        self.data[flat_index]
    }

    pub fn dot(&self, other: &Tensor) -> f64 {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            panic!("Dot product is only defined for 1D tensors");
        }
        if self.shape[0] != other.shape[0] {
            panic!("Tensors must have the same length for dot product");
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
}
