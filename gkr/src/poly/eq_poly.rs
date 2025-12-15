use field::Field;
use rayon::prelude::*;
use std::marker::PhantomData;

use crate::PAR_THRESHOLD;

/// The MLE of the function `eq(x, y)` on `{0,1}^2n` returning 1 if `x == y` and 0 otherwise
pub struct EqPoly<T>(PhantomData<T>);

impl<F: Field> EqPoly<F> {
    pub fn evals(r: &[F]) -> Vec<F> {
        Self::eval_with_scaling(r, F::ONE)
    }

    pub fn eval_with_scaling(r: &[F], scaling_factor: F) -> Vec<F> {
        if r.len() < PAR_THRESHOLD {
            Self::eval_serial(r, scaling_factor)
        } else {
            Self::eval_par(r, scaling_factor)
        }
    }

    pub fn eval_serial(r: &[F], scaling_factor: F) -> Vec<F> {
        let mut evals = vec![scaling_factor; 1 << r.len()];
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let s = evals[i / 2];
                evals[i] = s;
                evals[i].mul_assign(&r[j]);

                let tmp = evals[i];
                evals[i - 1] = s;
                evals[i - 1].sub_assign(&tmp);
            }
        }
        evals
    }

    pub fn eval_par(r: &[F], scaling_factor: F) -> Vec<F> {
        // TODO: for large r this can be slow
        // maybe use alloc_zero
        let mut evals = vec![scaling_factor; 1 << r.len()];
        let mut size = 1;

        for r in r.iter().rev() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x;
                    y.mul_assign(r);
                    x.sub_assign(y);
                });

            size *= 2;
        }

        evals
    }
}

#[cfg(test)]
mod tests {
    use field::{Mersenne31Field, PrimeField};
    use proptest::{collection::vec, prop_assert_eq, proptest};

    use super::*;

    type F = Mersenne31Field;

    fn to_index(bits: &[bool]) -> usize {
        bits.iter().fold(0, |acc, &val| acc * 2 + val as usize)
    }

    fn to_point(bits: &[bool]) -> Vec<F> {
        bits.iter().map(|&b| F::from_boolean(b)).collect()
    }

    #[test]
    fn eq_poly_on_small_hypercube() {
        proptest!(|(r in vec(proptest::bool::ANY, 0..20))| {
            let index = to_index(&r);
            let evals = EqPoly::evals(&to_point(&r));

            for i in 0..evals.len() {
                if i == index {
                    prop_assert_eq!(evals[i], F::ONE);
                } else {
                    prop_assert_eq!(evals[i], F::ZERO);
                }
            }
        })
    }

    #[test]
    fn eq_poly_on_large_hypercube() {
        fn test_indices(n: usize) -> Vec<usize> {
            vec![
                0,
                1,
                2,
                1 << (n / 4),
                1 << (n / 2),
                (1 << (n / 2)) + 1,
                (1 << n) - 2,
                (1 << n) - 1,
                0b10101010101010101010 & ((1 << n) - 1),
                0b01010101010101010101 & ((1 << n) - 1),
            ]
        }

        [20usize, 22, 24].into_par_iter().for_each(|n| {
            let test_indices = test_indices(n);

            test_indices.into_iter().for_each(|idx| {
                let bits: Vec<bool> = (0..n).map(|j| (idx >> j) & 1 == 1).collect();
                let point = to_point(&bits);
                let index = to_index(&bits);
                let evals = EqPoly::evals(&point);

                assert!(evals.into_par_iter().enumerate().all(|(i, x)| {
                    if i == index {
                        x == F::ONE
                    } else {
                        x == F::ZERO
                    }
                }));
            });
        });
    }

    #[test]
    fn eq_poly_serial_par_consistency() {
        proptest!(|(r in vec(0..u32::MAX, 0..16))| {
            let point: Vec<F> = r.into_iter().map(F::from_nonreduced_u32).collect();
            let serial = EqPoly::eval_serial(&point, F::ONE);
            let par = EqPoly::eval_par(&point, F::ONE);
            prop_assert_eq!(serial, par);
        })
    }
}
