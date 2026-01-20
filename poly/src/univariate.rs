use field::Field;
use rayon::prelude::*;
use std::ops::{Add, Mul};

use crate::PAR_THRESHOLD;

#[derive(Clone, Debug)]
pub struct UniPoly<F> {
    pub coeffs: Vec<F>,
}

impl<F: Field> UniPoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    pub fn degree(&self) -> Option<usize> {
        self.coeffs.iter().rposition(|c| !c.is_zero())
    }

    pub fn eval(&self, x: &F) -> F {
        self.coeffs.iter().rev().fold(F::ZERO, |mut acc, coeff| {
            acc.fused_mul_add_assign(&x, coeff);
            acc
        })
    }
}

// Scalar multiplication
impl<F: Field> Mul<&F> for UniPoly<F> {
    type Output = Self;

    fn mul(self, scalar: &F) -> Self::Output {
        let coeffs = self
            .coeffs
            .into_iter()
            .map(|mut c| {
                c.mul_assign(scalar);
                c
            })
            .collect();

        Self { coeffs }
    }
}

// Add two polynomials
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
