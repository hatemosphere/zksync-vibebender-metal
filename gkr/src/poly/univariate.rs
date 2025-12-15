use crate::PAR_THRESHOLD;

use super::{Group, Ring, Set, Polynomial, compressed_poly::CommpressedPoly};
use core::cmp::Ordering;
use core::ops::{Add, Mul, Neg, Sub};
use field::{Field, FieldExtension};
use rayon::prelude::*;

/// ax^2 + bx + c is represented as [c, b, a]
#[derive(Debug, Clone)]
pub struct UniPoly<F> {
    coeffs: Vec<F>,
}

impl<F: Field> UniPoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    pub fn compress(self) -> CommpressedPoly<F> {
        let coeffs_except_linear_term = [&self.coeffs[..1], &self.coeffs[2..]].concat();
        debug_assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
        CommpressedPoly::new(coeffs_except_linear_term)
    }
}

impl<F: Field> Mul<F> for UniPoly<F> {
    type Output = Self;

    fn mul(self, s: F) -> Self::Output {
        self * &s
    }
}

impl<'a, F: Field> Mul<&'a F> for UniPoly<F> {
    type Output = Self;

    fn mul(self, s: &'a F) -> Self::Output {
        Self {
            coeffs: self
                .coeffs
                .into_iter()
                .map(|mut x| {
                    x.mul_assign(s);
                    x
                })
                .collect(),
        }
    }
}

impl<F: Field> Polynomial<F> for UniPoly<F> {
    fn eval_at(&self, r: &[F]) -> F {
        self.coeffs.iter().rev().fold(F::ZERO, |mut acc, coeff| {
            acc.fused_mul_add_assign(&r[0], coeff);
            acc
        })
    }

    fn num_vars(&self) -> usize {
        1
    }

    fn degree(&self, _index: usize) -> usize {
        todo!()
    }

    fn sum_over_hypercube(&self) -> F {
        let mut sum = self.eval_at(&[F::ZERO]);
        sum.add_assign(&self.eval_at(&[F::ONE]));
        sum
    }

    fn lift<E: FieldExtension<F> + Field>(self) -> UniPoly<E> {
        UniPoly::<E>::new(self.coeffs.into_iter().map(E::from_base).collect())
    }
}

impl<F: Field> PartialEq for UniPoly<F> {
    fn eq(&self, other: &Self) -> bool {
        let a = self.coeffs.len();
        let b = other.coeffs.len();
        match a.cmp(&b) {
            Ordering::Less => {
                self.coeffs == other.coeffs[..a] && other.coeffs[a..].iter().all(|x| x.is_zero())
            }
            Ordering::Equal => self.coeffs == other.coeffs,
            Ordering::Greater => {
                self.coeffs[..b] == other.coeffs && self.coeffs[b..].iter().all(|x| x.is_zero())
            }
        }
    }
}

impl<F: Field> Default for UniPoly<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: Field> Eq for UniPoly<F> {}

impl<F: Field> Add for UniPoly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<'a, F: Field> Add<&'a Self> for UniPoly<F> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        let len = self.coeffs.len();
        if len < rhs.coeffs.len() {
            self.coeffs.resize(rhs.coeffs.len(), F::ZERO);
        }

        if len < PAR_THRESHOLD {
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(a, b)| {
                    a.add_assign(b);
                });
        } else {
            self.coeffs
                .par_iter_mut()
                .zip(rhs.coeffs.par_iter())
                .for_each(|(a, b)| {
                    a.add_assign(b);
                });
        }

        self
    }
}

impl<F: Field> Sub for UniPoly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<'a, F: Field> Sub<&'a Self> for UniPoly<F> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        let len = self.coeffs.len();
        if len < rhs.coeffs.len() {
            self.coeffs.resize(rhs.coeffs.len(), F::ZERO);
        }

        if len < PAR_THRESHOLD {
            self.coeffs
                .iter_mut()
                .zip(rhs.coeffs.iter())
                .for_each(|(a, b)| {
                    a.sub_assign(b);
                });
        } else {
            self.coeffs
                .par_iter_mut()
                .zip(rhs.coeffs.par_iter())
                .for_each(|(a, b)| {
                    a.sub_assign(b);
                });
        }

        self
    }
}

impl<F: Field> Neg for UniPoly<F> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        if self.coeffs.len() < PAR_THRESHOLD {
            self.coeffs.iter_mut().for_each(|x| {
                x.negate();
            });
        } else {
            self.coeffs.par_iter_mut().for_each(|x| {
                x.negate();
            });
        }

        self
    }
}

impl<F: Field> Mul for UniPoly<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, F: Field> Mul<&'a Self> for UniPoly<F> {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }

        let result_len = self.coeffs.len() + rhs.coeffs.len() - 1;

        if result_len < PAR_THRESHOLD {
            let mut coeffs = vec![F::ZERO; result_len];

            for (i, a) in self.coeffs.iter().enumerate() {
                for (j, b) in rhs.coeffs.iter().enumerate() {
                    let mut term = *a;
                    term.mul_assign(b);
                    coeffs[i + j].add_assign(&term);
                }
            }

            Self { coeffs }
        } else {
            let coeffs = (0..result_len)
                .into_par_iter()
                .map(|k| {
                    let mut sum = F::ZERO;

                    let start = k.saturating_sub(rhs.coeffs.len() - 1);
                    let end = (k + 1).min(self.coeffs.len());

                    for i in start..end {
                        let j = k - i;
                        let mut term = self.coeffs[i];
                        term.mul_assign(&rhs.coeffs[j]);
                        sum.add_assign(&term);
                    }

                    sum
                })
                .collect();

            Self { coeffs }
        }
    }
}

impl<F: Field> Set for UniPoly<F> {}
impl<F: Field> Group for UniPoly<F> {
    fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|x| x.is_zero())
    }
}
impl<F: Field> Ring for UniPoly<F> {
    fn one() -> Self {
        Self {
            coeffs: vec![F::ONE],
        }
    }

    fn is_one(&self) -> bool {
        let mut coeffs = self.coeffs.iter();
        coeffs.next().map(|x| x.is_one()).unwrap_or(false) && coeffs.all(|x| x.is_zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{test_group_prop, test_ring_prop};
    use field::base::Mersenne31Field;
    use proptest::{
        arbitrary::{any, Arbitrary, Mapped},
        prelude::Strategy,
    };

    type P = UniPoly<Mersenne31Field>;

    impl Arbitrary for P {
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            any::<Vec<u32>>().prop_map(|coeffs| Self {
                coeffs: coeffs
                    .into_iter()
                    .map(|c| Mersenne31Field::from_nonreduced_u32(c))
                    .collect(),
            })
        }

        type Strategy = Mapped<Vec<u32>, Self>;
    }

    #[test]
    fn prop_test() {
        test_group_prop::<P>();
        test_ring_prop::<P>();
    }
}
