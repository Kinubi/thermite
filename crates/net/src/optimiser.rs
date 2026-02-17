pub use crate::optimisers::Adam;
pub use crate::module::Module;

pub trait Optimiser {
    fn from_model<M: Module + ?Sized>(model: &M) -> Self where Self: Sized;

    fn zero_grad<M: Module + ?Sized>(&mut self, model: &mut M);

    fn step<M: Module + ?Sized>(&mut self, model: &mut M);
}
