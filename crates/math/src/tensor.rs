use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub trait TensorNest {
    fn infer_shape(&self) -> Vec<usize>;
    fn flatten_into(&self, out: &mut Vec<f64>);
}

impl TensorNest for f64 {
    fn infer_shape(&self) -> Vec<usize> {
        Vec::new()
    }

    fn flatten_into(&self, out: &mut Vec<f64>) {
        out.push(*self);
    }
}

impl<T: TensorNest> TensorNest for Vec<T> {
    fn infer_shape(&self) -> Vec<usize> {
        if self.is_empty() {
            return vec![0];
        }

        let child_shape = self[0].infer_shape();
        for item in &self[1..] {
            let item_shape = item.infer_shape();
            if item_shape != child_shape {
                panic!("Ragged nested Vec cannot be converted into a Tensor");
            }
        }

        let mut shape = Vec::with_capacity(1 + child_shape.len());
        shape.push(self.len());
        shape.extend(child_shape);
        shape
    }

    fn flatten_into(&self, out: &mut Vec<f64>) {
        for item in self {
            item.flatten_into(out);
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor").field("data", &self.data).field("shape", &self.shape).finish()
    }
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let size = Self::numel_of(&shape);
        if size != data.len() {
            panic!("Shape {:?} implies {} elements, but data has {}", shape, size, data.len());
        }
        let strides = Self::strides_of(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn scalar(value: f64) -> Self {
        Self::new(vec![value], Vec::new())
    }

    pub fn from_nested<T: TensorNest>(nested: T) -> Self {
        let shape = nested.infer_shape();
        let mut data = Vec::with_capacity(Self::numel_of(&shape));
        nested.flatten_into(&mut data);
        Self::new(data, shape)
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self::new(data, vec![len])
    }

    pub fn from_vec2(rows: Vec<Vec<f64>>) -> Self {
        let row_count = rows.len();
        let col_count = rows
            .first()
            .map(|r| r.len())
            .unwrap_or(0);

        for row in &rows {
            if row.len() != col_count {
                panic!("All rows must have the same length");
            }
        }

        let mut data = Vec::with_capacity(row_count * col_count);
        for row in rows {
            data.extend(row);
        }

        Self::new(data, vec![row_count, col_count])
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = Self::numel_of(&shape);
        Self::new(vec![0.0; size], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size = Self::numel_of(&shape);
        Self::new(vec![1.0; size], shape)
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let new_size = Self::numel_of(&new_shape);
        if new_size != self.data.len() {
            panic!("New shape must have the same number of elements as the original shape");
        }
        self.shape = new_shape;
        self.strides = Self::strides_of(&self.shape);
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, indices: Vec<usize>) -> f64 {
        *self.get_ref(&indices)
    }

    pub fn get_ref(&self, indices: &[usize]) -> &f64 {
        let flat_index = self.flat_index(indices);
        &self.data[flat_index]
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> &mut f64 {
        let flat_index = self.flat_index(indices);
        &mut self.data[flat_index]
    }

    pub fn indices(&self) -> Indices {
        Indices::new(self.shape.clone())
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.data.iter_mut()
    }

    pub fn iter_indexed(&self) -> IndexedIter<'_> {
        IndexedIter {
            indices: self.indices(),
            data_iter: self.data.iter(),
        }
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

    pub fn data_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn shape_slice(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: &self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
        }
    }

    pub fn iter_flat_indexed(&self) -> impl Iterator<Item = (usize, &f64)> {
        self.data.iter().enumerate()
    }

    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> TensorView<'_> {
        self.view().narrow(axis, start, len)
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Number of indices must match the number of dimensions");
        }

        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            let dim = self.shape[i];
            if idx >= dim {
                panic!("Index out of bounds");
            }
            flat_index += idx * self.strides[i];
        }
        flat_index
    }

    fn numel_of(shape: &[usize]) -> usize {
        shape.iter().copied().product()
    }

    fn strides_of(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut acc = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }
}

#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    data: &'a [f64],
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<'a> TensorView<'a> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }

    pub fn get_ref(&self, indices: &[usize]) -> &'a f64 {
        let flat = self.flat_index(indices);
        &self.data[flat]
    }

    pub fn indices(&self) -> Indices {
        Indices::new(self.shape.clone())
    }

    pub fn iter(&self) -> TensorViewIter<'a> {
        TensorViewIter {
            view: self.clone(),
            indices: self.indices(),
        }
    }

    pub fn iter_fast(&self) -> TensorViewIterFast<'a> {
        TensorViewIterFast::new(self.clone())
    }

    pub fn iter_indexed(&self) -> TensorViewIndexedIter<'a> {
        TensorViewIndexedIter {
            view: self.clone(),
            indices: self.indices(),
        }
    }

    pub fn iter_flat_indexed(&self) -> TensorViewFlatIndexedIter<'a> {
        TensorViewFlatIndexedIter::new(self.clone())
    }

    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> TensorView<'a> {
        if axis >= self.shape.len() {
            panic!("Axis out of bounds");
        }
        if start > self.shape[axis] || start + len > self.shape[axis] {
            panic!("Slice out of bounds");
        }

        let mut shape = self.shape.clone();
        shape[axis] = len;
        let offset = self.offset + start * self.strides[axis];

        TensorView {
            data: self.data,
            shape,
            strides: self.strides.clone(),
            offset,
        }
    }

    pub fn transpose2(&self) -> TensorView<'a> {
        if self.shape.len() != 2 {
            panic!("transpose2 is only defined for rank-2 tensors");
        }
        TensorView {
            data: self.data,
            shape: vec![self.shape[1], self.shape[0]],
            strides: vec![self.strides[1], self.strides[0]],
            offset: self.offset,
        }
    }

    pub fn to_owned(&self) -> Tensor {
        let mut out = Vec::with_capacity(self.numel());
        for idx in self.indices() {
            out.push(*self.get_ref(&idx));
        }
        Tensor::new(out, self.shape.clone())
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        if indices.len() != self.shape.len() {
            panic!("Number of indices must match the number of dimensions");
        }

        let mut flat_index = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            let dim = self.shape[i];
            if idx >= dim {
                panic!("Index out of bounds");
            }
            flat_index += idx * self.strides[i];
        }
        flat_index
    }
}

pub struct TensorViewIter<'a> {
    view: TensorView<'a>,
    indices: Indices,
}

impl<'a> Iterator for TensorViewIter<'a> {
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.indices.next()?;
        Some(self.view.get_ref(&idx))
    }
}

pub struct TensorViewIndexedIter<'a> {
    view: TensorView<'a>,
    indices: Indices,
}

pub struct TensorViewIterFast<'a> {
    view: TensorView<'a>,
    current: Vec<usize>,
    done: bool,
}

impl<'a> TensorViewIterFast<'a> {
    fn new(view: TensorView<'a>) -> Self {
        if view.shape.iter().any(|&d| d == 0) {
            return Self {
                view,
                current: Vec::new(),
                done: true,
            };
        }

        Self {
            current: vec![0; view.shape.len()],
            view,
            done: false,
        }
    }
}

impl<'a> Iterator for TensorViewIterFast<'a> {
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let out = self.view.get_ref(&self.current);

        if self.view.shape.is_empty() {
            self.done = true;
            return Some(out);
        }

        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.view.shape[i] {
                break;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(out)
    }
}

pub struct TensorViewFlatIndexedIter<'a> {
    view: TensorView<'a>,
    current: Vec<usize>,
    done: bool,
    logical_index: usize,
}

impl<'a> TensorViewFlatIndexedIter<'a> {
    fn new(view: TensorView<'a>) -> Self {
        if view.shape.iter().any(|&d| d == 0) {
            return Self {
                view,
                current: Vec::new(),
                done: true,
                logical_index: 0,
            };
        }

        Self {
            current: vec![0; view.shape.len()],
            view,
            done: false,
            logical_index: 0,
        }
    }
}

impl<'a> Iterator for TensorViewFlatIndexedIter<'a> {
    type Item = (usize, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let out_val = self.view.get_ref(&self.current);
        let out_idx = self.logical_index;
        self.logical_index += 1;

        if self.view.shape.is_empty() {
            self.done = true;
            return Some((out_idx, out_val));
        }

        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.view.shape[i] {
                break;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some((out_idx, out_val))
    }
}

fn assert_same_shape(lhs: &Tensor, rhs: &Tensor) {
    if lhs.shape != rhs.shape {
        panic!("Shape mismatch: {:?} vs {:?}", lhs.shape, rhs.shape);
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        assert_same_shape(self, rhs);
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b;
        }
    }
}

impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: &Tensor) {
        assert_same_shape(self, rhs);
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b;
        }
    }
}

impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: &Tensor) {
        assert_same_shape(self, rhs);
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a *= b;
        }
    }
}

impl DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, rhs: &Tensor) {
        assert_same_shape(self, rhs);
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a /= b;
        }
    }
}

impl AddAssign<f64> for Tensor {
    fn add_assign(&mut self, rhs: f64) {
        for a in &mut self.data {
            *a += rhs;
        }
    }
}

impl SubAssign<f64> for Tensor {
    fn sub_assign(&mut self, rhs: f64) {
        for a in &mut self.data {
            *a -= rhs;
        }
    }
}

impl MulAssign<f64> for Tensor {
    fn mul_assign(&mut self, rhs: f64) {
        for a in &mut self.data {
            *a *= rhs;
        }
    }
}

impl DivAssign<f64> for Tensor {
    fn div_assign(&mut self, rhs: f64) {
        for a in &mut self.data {
            *a /= rhs;
        }
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: &'b Tensor) -> Tensor {
        assert_same_shape(self, rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &'b Tensor) -> Tensor {
        assert_same_shape(self, rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &'b Tensor) -> Tensor {
        assert_same_shape(self, rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl<'a, 'b> Div<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn div(self, rhs: &'b Tensor) -> Tensor {
        assert_same_shape(self, rhs);
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a / b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(mut self, rhs: Tensor) -> Tensor {
        self += &rhs;
        self
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(mut self, rhs: Tensor) -> Tensor {
        self -= &rhs;
        self
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(mut self, rhs: Tensor) -> Tensor {
        self *= &rhs;
        self
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(mut self, rhs: Tensor) -> Tensor {
        self /= &rhs;
        self
    }
}

impl<'a> Add<f64> for &'a Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Tensor {
        let data = self.data.iter().map(|a| a + rhs).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl<'a> Sub<f64> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Tensor {
        let data = self.data.iter().map(|a| a - rhs).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl<'a> Mul<f64> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Tensor {
        let data = self.data.iter().map(|a| a * rhs).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl<'a> Div<f64> for &'a Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Tensor {
        let data = self.data.iter().map(|a| a / rhs).collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(mut self, rhs: f64) -> Tensor {
        self += rhs;
        self
    }
}

impl Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(mut self, rhs: f64) -> Tensor {
        self -= rhs;
        self
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(mut self, rhs: f64) -> Tensor {
        self *= rhs;
        self
    }
}

impl Div<f64> for Tensor {
    type Output = Tensor;

    fn div(mut self, rhs: f64) -> Tensor {
        self /= rhs;
        self
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(mut self) -> Tensor {
        for a in &mut self.data {
            *a = -*a;
        }
        self
    }
}

impl<'a> Iterator for TensorViewIndexedIter<'a> {
    type Item = (Vec<usize>, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.indices.next()?;
        let val = self.view.get_ref(&idx);
        Some((idx, val))
    }
}

#[derive(Debug, Clone)]
pub struct Indices {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl Indices {
    pub fn new(shape: Vec<usize>) -> Self {
        if shape.iter().any(|&d| d == 0) {
            return Self {
                shape,
                current: Vec::new(),
                done: true,
            };
        }

        let current = vec![0; shape.len()];
        Self {
            shape,
            current,
            done: false,
        }
    }
}

impl Iterator for Indices {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let out = self.current.clone();

        if self.shape.is_empty() {
            self.done = true;
            return Some(out);
        }

        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                break;
            }
            self.current[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(out)
    }
}

pub struct IndexedIter<'a> {
    indices: Indices,
    data_iter: std::slice::Iter<'a, f64>,
}

impl<'a> Iterator for IndexedIter<'a> {
    type Item = (Vec<usize>, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.indices.next()?;
        let val = self.data_iter.next()?;
        Some((idx, val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_2d_row_major() {
        let t = Tensor::zeros(vec![2, 3]);
        let idx: Vec<Vec<usize>> = t.indices().collect();
        assert_eq!(
            idx,
            vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![1, 0], vec![1, 1], vec![1, 2]]
        );
    }

    #[test]
    fn iter_indexed_matches_data_order() {
        let t = Tensor::new(vec![10.0, 11.0, 12.0, 13.0], vec![2, 2]);
        let got: Vec<(Vec<usize>, f64)> = t
            .iter_indexed()
            .map(|(i, v)| (i, *v))
            .collect();
        assert_eq!(
            got,
            vec![(vec![0, 0], 10.0), (vec![0, 1], 11.0), (vec![1, 0], 12.0), (vec![1, 1], 13.0)]
        );
    }

    #[test]
    fn from_nested_3d_infers_shape_and_flattens_row_major() {
        let nested = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]]
        ];
        let t = Tensor::from_nested(nested);
        assert_eq!(t.shape(), &vec![2, 2, 2]);
        assert_eq!(t[[0, 0, 0]], 1.0);
        assert_eq!(t[[0, 1, 1]], 4.0);
        assert_eq!(t[[1, 0, 1]], 6.0);
        assert_eq!(t[[1, 1, 0]], 7.0);
    }

    #[test]
    fn from_nested_rejects_ragged() {
        let ragged = vec![vec![1.0, 2.0], vec![3.0]];
        let result = std::panic::catch_unwind(|| Tensor::from_nested(ragged));
        assert!(result.is_err());
    }

    #[test]
    fn view_narrow_and_transpose2() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = t.view();
        assert_eq!(*v.get_ref(&[0, 2]), 3.0);

        let row1 = v.narrow(0, 1, 1);
        assert_eq!(row1.shape(), &[1, 3]);
        assert_eq!(*row1.get_ref(&[0, 1]), 5.0);

        let vt = v.transpose2();
        assert_eq!(vt.shape(), &[3, 2]);
        assert_eq!(*vt.get_ref(&[2, 1]), 6.0);
    }

    #[test]
    fn view_iter_matches_view_indexing_order() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vt = t.view().transpose2();

        // vt is shape [3,2], elements by (i,j):
        // (0,0)=1 (0,1)=4 (1,0)=2 (1,1)=5 (2,0)=3 (2,1)=6
        let got: Vec<f64> = vt.iter().copied().collect();
        assert_eq!(got, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn view_iter_fast_matches_iter() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = t.view().transpose2().narrow(0, 0, 2);
        let slow: Vec<f64> = v.iter().copied().collect();
        let fast: Vec<f64> = v.iter_fast().copied().collect();
        assert_eq!(slow, fast);
    }

    #[test]
    fn tensor_ops_elementwise_and_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

        let c = &a + &b;
        assert_eq!(c.data_slice(), &[11.0, 22.0, 33.0, 44.0]);

        let d = &b - &a;
        assert_eq!(d.data_slice(), &[9.0, 18.0, 27.0, 36.0]);

        let e = &a * &b;
        assert_eq!(e.data_slice(), &[10.0, 40.0, 90.0, 160.0]);

        let f = &a * 2.0;
        assert_eq!(f.data_slice(), &[2.0, 4.0, 6.0, 8.0]);

        let g = a.clone() + 1.0;
        assert_eq!(g.data_slice(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn view_flat_indexed_counts_in_logical_order() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = t.view().transpose2();
        let got: Vec<(usize, f64)> = v.iter_flat_indexed().map(|(i, v)| (i, *v)).collect();
        assert_eq!(
            got,
            vec![(0, 1.0), (1, 4.0), (2, 2.0), (3, 5.0), (4, 3.0), (5, 6.0)]
        );
    }
}

impl<'a> Index<&[usize]> for TensorView<'a> {
    type Output = f64;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        self.get_ref(indices)
    }
}

impl<'a, const N: usize> Index<[usize; N]> for TensorView<'a> {
    type Output = f64;

    fn index(&self, indices: [usize; N]) -> &Self::Output {
        self.get_ref(&indices)
    }
}

impl Index<usize> for Tensor {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Index<(usize, usize)> for Tensor {
    type Output = f64;
    fn index(&self, indices: (usize, usize)) -> &Self::Output {
        let (i, j) = indices;
        if self.shape.len() != 2 {
            panic!("Indexing with two indices is only defined for 2D tensors");
        }
        if i >= self.shape[0] || j >= self.shape[1] {
            panic!("Index out of bounds");
        }
        &self.data[i * self.shape[1] + j]
    }
}

impl Index<&[usize]> for Tensor {
    type Output = f64;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        self.get_ref(indices)
    }
}

impl IndexMut<&[usize]> for Tensor {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        self.get_mut(indices)
    }
}

impl<const N: usize> Index<[usize; N]> for Tensor {
    type Output = f64;

    fn index(&self, indices: [usize; N]) -> &Self::Output {
        self.get_ref(&indices)
    }
}

impl<const N: usize> IndexMut<[usize; N]> for Tensor {
    fn index_mut(&mut self, indices: [usize; N]) -> &mut Self::Output {
        self.get_mut(&indices)
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, indices: (usize, usize)) -> &mut Self::Output {
        let (i, j) = indices;
        if self.shape.len() != 2 {
            panic!("Indexing with two indices is only defined for 2D tensors");
        }
        if i >= self.shape[0] || j >= self.shape[1] {
            panic!("Index out of bounds");
        }
        &mut self.data[i * self.shape[1] + j]
    }
}
