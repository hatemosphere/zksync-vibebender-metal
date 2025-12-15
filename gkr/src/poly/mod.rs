use core::fmt::Debug;
use core::ops::{Add, Mul, Neg, Sub};
use field::{Field, FieldExtension};

mod eq_poly;
mod multilinear;
// mod multivariate;
// mod sparse_multilinear;
mod univariate;
mod compressed_poly;

pub use multilinear::MultilinearPoly;
pub use univariate::UniPoly;
pub use compressed_poly::CommpressedPoly;

#[cfg(test)]
use proptest::{arbitrary::Arbitrary, prop_assert_eq, proptest};

use crate::transcript::Transcript;

pub trait Set: Sized + Eq + Debug + Clone + Default {}

pub trait Group:
    Set
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
}

pub trait Ring: Group + Mul<Output = Self> + for<'a> Mul<&'a Self, Output = Self> {
    fn one() -> Self;
    fn is_one(&self) -> bool;
}

pub trait Polynomial<F: Field>:
    Group + Mul<F, Output = Self> + for<'a> Mul<&'a F, Output = Self>
{
    fn eval_at(&self, r: &[F]) -> F;
    fn num_vars(&self) -> usize;
    fn degree(&self, index: usize) -> usize;
    fn sum_over_hypercube(&self) -> F;
    fn lift<E: FieldExtension<F> + Field>(self) -> impl Polynomial<E>;
}

pub trait AppendToTranscript<F, E, T> 
where
    F: Field,
    E: FieldExtension<F> + Field,
    T: Transcript<F, Extension = E>
{
    fn append_to_transcript(&self, transcript: &mut T);
}

pub trait SumcheckPoly<F: Field>: Polynomial<F> {
    /// binds first `r.len()` variables
    fn partial_eval(&self, r: &[F]) -> Self;
    /// sums over the first `k` variables
    fn partial_sum(&self, k: usize) -> Self;
}

#[cfg(test)]
fn test_group_prop<T: 'static + Group + Arbitrary>() {
    proptest!(|(a: T, b: T, c: T)| {
        prop_assert_eq!(a.clone() + T::zero(), a.clone());
        prop_assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());
        prop_assert_eq!(a.clone() - a.clone(), T::zero());
        prop_assert_eq!(a.clone() + (b.clone() + c.clone()), (a + b) + c);
    })
}

#[cfg(test)]
fn test_ring_prop<T: 'static + Ring + Arbitrary>() {
    proptest!(|(a: T, b: T, c: T)| {
        prop_assert_eq!(T::one() * a.clone(), a.clone());
        prop_assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());
        prop_assert_eq!(a.clone() * (b.clone() * c.clone()), (a.clone() * b.clone()) * c.clone());
        prop_assert_eq!(a.clone() * (b.clone() + c.clone()), a.clone() * b + a * c);
    })
}
