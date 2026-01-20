// based on https://github.com/a16z/jolt/blob/main/jolt-core/src/poly/eq_poly.rs#
use core::marker::PhantomData;

use field::Field;
use rayon::prelude::*;

use crate::{MultiLinearPoly, PAR_THRESHOLD};
pub struct EqPoly<F>(PhantomData<F>);

impl<F: Field> EqPoly<F> {
    /// returns eq(r, x) as a multilinear polynomial in x
    pub fn eval(r: &[F]) -> MultiLinearPoly<F> {
        Self::eval_with_scaling(r, F::ONE)
    }

    /// returns scaling_factor * eq(r, x) as a multilinear polynomial in x
    pub fn eval_with_scaling(r: &[F], scaling_factor: F) -> MultiLinearPoly<F> {
        let evals = if r.len() < PAR_THRESHOLD {
            Self::eval_serial(r, scaling_factor)
        } else {
            Self::eval_par(r, scaling_factor)
        };

        MultiLinearPoly::from_evals(evals)
    }

    fn eval_serial(r: &[F], scaling_factor: F) -> Vec<F> {
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

    fn eval_par(r: &[F], scaling_factor: F) -> Vec<F> {
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

    // cached version of `Self::eval`
    // returns result where result[j] = eq(r[..j], x)
    pub fn eval_cached(r: &[F]) -> Vec<MultiLinearPoly<F>> {
        Self::eval_cached_with_scaling(r, F::ONE)
    }

    pub fn eval_cached_with_scaling(r: &[F], scaling_factor: F) -> Vec<MultiLinearPoly<F>> {
        let mut evals = Vec::with_capacity(r.len() + 1);
        evals.push(vec![scaling_factor]);

        let mut r_iter = r.iter().enumerate();

        for (j, r_val) in r_iter.by_ref().take(PAR_THRESHOLD.ilog2() as usize) {
            evals.push(Self::next_layer_serial(&evals[j], r_val));
        }

        for (j, r_val) in r_iter {
            evals.push(Self::next_layer_par(&evals[j], r_val));
        }

        evals.into_iter().map(MultiLinearPoly::from_evals).collect()
    }

    fn next_layer_serial(prev_layer: &Vec<F>, r_val: &F) -> Vec<F> {
        prev_layer
            .iter()
            .flat_map(|&scalar| {
                let mut right = scalar;
                right.mul_assign(&r_val);

                let mut left = scalar;
                left.sub_assign(&right);

                [left, right]
            })
            .collect()
    }

    fn next_layer_par(prev_layer: &Vec<F>, r_val: &F) -> Vec<F> {
        let next_len = 2 * prev_layer.len();
        let mut next_layer = Vec::with_capacity(next_len);

        // Safety: we overwrite each value before we read it
        unsafe {
            next_layer.set_len(next_len);
        }

        next_layer
            .par_chunks_mut(2)
            .zip(prev_layer.par_iter())
            .for_each(|(chunck, &scalar)| {
                let mut right = scalar;
                right.mul_assign(&r_val);

                let mut left = scalar;
                left.sub_assign(&right);

                chunck[0] = left;
                chunck[1] = right;
            });

        next_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::{Field, Mersenne31Field};
    use proptest::prelude::*;

    type F = Mersenne31Field;

    // implementation from https://github.com/a16z/jolt/blob/main/jolt-core/src/poly/eq_poly.rs#L187
    fn jolt_reference_eval_cached(r: &[F], scaling_factor: F) -> Vec<Vec<F>> {
        let mut evals: Vec<Vec<F>> = (0..r.len() + 1)
            .map(|i| vec![scaling_factor; 1 << i])
            .collect();

        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[j][i / 2];
                let mut right = scalar;
                right.mul_assign(&r[j]);

                let mut left = scalar;
                left.sub_assign(&right);

                evals[j + 1][i] = right;
                evals[j + 1][i - 1] = left;
            }
        }
        evals
    }

    proptest! {
        #[test]
        fn test_next_layer_equivalence(
            prev_layer: Vec<u32>,
            r_val in any::<u32>()
        ) {

            let prev_layer = prev_layer.into_iter().map(F::from_nonreduced_u32).collect();
            let r_val = F::from_nonreduced_u32(r_val);

            let serial_res = EqPoly::next_layer_serial(&prev_layer, &r_val);
            let par_res = EqPoly::next_layer_par(&prev_layer, &r_val);

            prop_assert_eq!(serial_res, par_res);
        }

        #[test]
        fn test_eval_cached_matches_jolt(
            r in prop::collection::vec(any::<u32>(), 0..20),
            scaling_factor: u32
        ) {
            let r: Vec<F> = r.into_iter().map(F::from_nonreduced_u32).collect();
            let scaling_factor = F::from_nonreduced_u32(scaling_factor);


            let optimized_polys = EqPoly::eval_cached_with_scaling(&r, scaling_factor);

            let optimized_vecs: Vec<Vec<F>> = optimized_polys
                .into_iter()
                .map(|p| p.evals().to_vec())
                .collect();

            let reference_vecs = jolt_reference_eval_cached(&r, scaling_factor);

            prop_assert_eq!(optimized_vecs, reference_vecs);
        }
    }
}
