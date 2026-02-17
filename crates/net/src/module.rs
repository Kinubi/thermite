pub use crate::layer::Layer;

pub trait Module: Layer {
    fn train(&mut self) {}

    fn eval(&mut self) {}
}

impl<T: Layer + ?Sized> Module for T {}
