use core::{
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
};

use crate::poly::{MultilinearPoly, Polynomial, UniPoly};

use super::{Group, Ring, Set};
use field::Field;

#[derive(Clone, Debug)]
pub struct MultiPoly<T>(PhantomData<T>);

impl<F: Field> Default for MultiPoly<F> {
    fn default() -> Self {
        todo!()
    }
}

impl<F: Field> PartialEq for MultiPoly<F> {
    fn eq(&self, _other: &Self) -> bool {
        todo!()
    }
}

impl<F: Field> Eq for MultiPoly<F> {}
impl<F: Field> Set for MultiPoly<F> {}

impl<F: Field> Add for MultiPoly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<F: Field> Add<&Self> for MultiPoly<F> {
    type Output = Self;

    fn add(self, _rhs: &Self) -> Self::Output {
        todo!()
    }
}

impl<F: Field> Neg for MultiPoly<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        todo!()
    }
}

impl<F: Field> Sub for MultiPoly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<F: Field> Sub<&Self> for MultiPoly<F> {
    type Output = Self;

    fn sub(self, _rhs: &Self) -> Self::Output {
        todo!()
    }
}

impl<F: Field> Group for MultiPoly<F> {
    fn zero() -> Self {
        todo!()
    }

    fn is_zero(&self) -> bool {
        todo!()
    }
}

impl<F: Field> Mul for MultiPoly<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<F: Field> Mul<&Self> for MultiPoly<F> {
    type Output = Self;

    fn mul(self, _rhs: &Self) -> Self::Output {
        todo!()
    }
}

impl<F: Field> Ring for MultiPoly<F> {
    fn one() -> Self {
        todo!()
    }

    fn is_one(&self) -> bool {
        todo!()
    }
}

impl<F: Field> Mul<F> for MultiPoly<F> {
    type Output = Self;

    fn mul(self, _rhs: F) -> Self::Output {
        todo!()
    }
}

impl<'a, F: Field> Mul<&'a F> for MultiPoly<F> {
    type Output = Self;

    fn mul(self, _rhs: &F) -> Self::Output {
        todo!()
    }
}

impl<F: Field> Polynomial<F> for MultiPoly<F> {
    fn eval_at(&self, _r: &[F]) -> F {
        todo!()
    }

    fn num_vars(&self) -> usize {
        todo!()
    }

    fn degree(&self, _index: usize) -> usize {
        todo!()
    }

    fn sum_over_hypercube(&self) -> F {
        todo!()
    }

    fn lift<E: field::FieldExtension<F> + Field>(self) -> impl Polynomial<E> {
        todo!();
        MultiPoly::<E>(PhantomData)
    }
}

impl<F: Field> From<UniPoly<F>> for MultiPoly<F> {
    fn from(_value: UniPoly<F>) -> Self {
        todo!()
    }
}

impl<F: Field> From<MultilinearPoly<F>> for MultiPoly<F> {
    fn from(_value: MultilinearPoly<F>) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{test_group_prop, test_ring_prop};
    use field::base::Mersenne31Field;
    use proptest::arbitrary::{Arbitrary, Mapped};

    type P = MultiPoly<Mersenne31Field>;

    impl Arbitrary for P {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            todo!()
        }

        // placeholder for now
        type Strategy = Mapped<u32, P>;
    }

    #[ignore = "unimplemented"]
    #[test]
    fn prop_test() {
        test_group_prop::<P>();
        test_ring_prop::<P>();
    }
}
