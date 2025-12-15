use super::{eq_poly::EqPoly, Group, Polynomial, Set};
use core::ops::{Add, Mul, Neg, Sub};
use field::{Field, FieldExtension};
use std::collections::HashMap;

/// Sparse multilinear polynomial storing only non-zero evaluations
/// on the boolean hypercube {0,1}^n
pub struct SparseMultilinearPoly<F> {
    /// Map from hypercube index to evaluation
    /// Index encoding: for point [x0, x1, ..., x_{n-1}],
    /// index = x_{n-1} + 2*x_{n-2} + ... + 2^{n-1}*x_0 (MSB first)
    evals: HashMap<usize, F>,

    /// Number of variables
    num_vars: usize,
}

impl<F: Field> SparseMultilinearPoly<F> {
    /// Create new sparse MLE with specified number of variables
    pub fn new(num_vars: usize) -> Self {
        Self {
            evals: HashMap::new(),
            num_vars,
        }
    }

    /// Create from dense evaluations (filters zeros)
    pub fn from_dense(evals: Vec<F>, num_vars: usize) -> Self {
        let mut sparse_evals = HashMap::new();
        for (idx, &val) in evals.iter().enumerate() {
            if !val.is_zero() {
                sparse_evals.insert(idx, val);
            }
        }
        Self {
            evals: sparse_evals,
            num_vars,
        }
    }

    /// Create from iterator of (index, value) pairs
    pub fn from_iter<I>(iter: I, num_vars: usize) -> Self
    where
        I: IntoIterator<Item = (usize, F)>,
    {
        Self {
            evals: iter.into_iter().collect(),
            num_vars,
        }
    }

    /// Insert or update evaluation at boolean index
    pub fn insert(&mut self, index: usize, value: F) {
        if !value.is_zero() {
            self.evals.insert(index, value);
        } else {
            self.evals.remove(&index);
        }
    }

    /// Get evaluation at boolean index
    pub fn get(&self, index: usize) -> F {
        self.evals.get(&index).copied().unwrap_or(F::ZERO)
    }

    /// Number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.evals.len()
    }

    /// Sparsity ratio (nnz / total_size)
    pub fn density(&self) -> f64 {
        self.nnz() as f64 / (1 << self.num_vars) as f64
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Vec<F> {
        let size = 1 << self.num_vars;
        let mut dense = vec![F::ZERO; size];
        for (&idx, &val) in &self.evals {
            dense[idx] = val;
        }
        dense
    }

    /// Convert boolean vector to array index
    /// [b0, b1, b2] -> b2 + 2*b1 + 4*b0 (reversed bit order, LSB is rightmost)
    fn bool_vec_to_index(bools: &[F]) -> usize {
        let mut idx = 0;
        for (i, &b) in bools.iter().enumerate() {
            if b == F::ONE {
                idx |= 1 << (bools.len() - 1 - i);
            }
        }
        idx
    }
}

impl<F: Field> Polynomial<F> for SparseMultilinearPoly<F> {
    fn eval_at(&self, r: &[F]) -> F {
        assert_eq!(r.len(), self.num_vars);

        // Check if r is boolean (all coords 0 or 1)
        let is_boolean = r.iter().all(|&x| x == F::ZERO || x == F::ONE);

        if is_boolean {
            // Direct lookup
            let idx = Self::bool_vec_to_index(r);
            self.get(idx)
        } else {
            // Lagrange interpolation using eq polynomials
            // f(r) = Σ_{x ∈ {0,1}^n} f(x) · eq(r, x)
            //      = Σ_{x: f(x) ≠ 0} f(x) · eq(r, x)  (sparse)

            let eq_evals = EqPoly::<F>::evals(r);
            let mut sum = F::ZERO;

            for (&idx, &val) in &self.evals {
                let mut contrib = val;
                contrib.mul_assign(&eq_evals[idx]);
                sum.add_assign(&contrib);
            }

            sum
        }
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self, _index: usize) -> usize {
        1 // Multilinear
    }

    fn sum_over_hypercube(&self) -> F {
        self.evals.values().fold(F::ZERO, |mut acc, &val| {
            acc.add_assign(&val);
            acc
        })
    }

    fn lift<E: FieldExtension<F> + Field>(self) -> SparseMultilinearPoly<E> {
        SparseMultilinearPoly::<E>::from_iter(
            self.evals
                .into_iter()
                .map(|(idx, val)| (idx, E::from_base(val))),
            self.num_vars,
        )
    }
}

// Group and Mul trait implementations required by Polynomial trait

impl<F: Field> Set for SparseMultilinearPoly<F> {}

impl<F: Field> Group for SparseMultilinearPoly<F> {
    fn zero() -> Self {
        Self {
            evals: HashMap::new(),
            num_vars: 0,
        }
    }

    fn is_zero(&self) -> bool {
        self.evals.is_empty()
    }
}

impl<F: Field> Add for SparseMultilinearPoly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<F: Field> Add<&Self> for SparseMultilinearPoly<F> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        assert_eq!(self.num_vars, rhs.num_vars);

        for (&idx, &val) in &rhs.evals {
            let current = self.get(idx);
            let mut new_val = current;
            new_val.add_assign(&val);
            self.insert(idx, new_val);
        }

        self
    }
}

impl<F: Field> Sub for SparseMultilinearPoly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<F: Field> Sub<&Self> for SparseMultilinearPoly<F> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        assert_eq!(self.num_vars, rhs.num_vars);

        for (&idx, &val) in &rhs.evals {
            let current = self.get(idx);
            let mut new_val = current;
            new_val.sub_assign(&val);
            self.insert(idx, new_val);
        }

        self
    }
}

impl<F: Field> Neg for SparseMultilinearPoly<F> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for val in self.evals.values_mut() {
            val.negate();
        }
        self
    }
}

impl<F: Field> Mul<F> for SparseMultilinearPoly<F> {
    type Output = Self;

    fn mul(self, s: F) -> Self::Output {
        self * &s
    }
}

impl<'a, F: Field> Mul<&'a F> for SparseMultilinearPoly<F> {
    type Output = Self;

    fn mul(mut self, s: &F) -> Self::Output {
        for val in self.evals.values_mut() {
            val.mul_assign(s);
        }
        self
    }
}

impl<F: Field> PartialEq for SparseMultilinearPoly<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.num_vars != other.num_vars {
            return false;
        }

        // Check that all non-zero entries match
        for (&idx, &val) in &self.evals {
            if other.get(idx) != val {
                return false;
            }
        }

        // Check that other has no non-zero entries we don't have
        for (&idx, &val) in &other.evals {
            if !val.is_zero() && self.get(idx) != val {
                return false;
            }
        }

        true
    }
}

impl<F: Field> Eq for SparseMultilinearPoly<F> {}

impl<F: Field> Clone for SparseMultilinearPoly<F> {
    fn clone(&self) -> Self {
        Self {
            evals: self.evals.clone(),
            num_vars: self.num_vars,
        }
    }
}

impl<F: Field> Default for SparseMultilinearPoly<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: Field> core::fmt::Debug for SparseMultilinearPoly<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SparseMultilinearPoly")
            .field("num_vars", &self.num_vars)
            .field("nnz", &self.nnz())
            .field("density", &self.density())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::MultilinearPoly;
    use field::{Mersenne31Complex as E, Mersenne31Field as F};

    #[test]
    fn test_sparse_vs_dense_evaluations() {
        // Create sparse with known non-zeros
        let mut sparse = SparseMultilinearPoly::new(3);
        sparse.insert(0, F::from_nonreduced_u32(1)); // [0,0,0]
        sparse.insert(5, F::from_nonreduced_u32(7)); // [1,0,1]

        // Create equivalent dense
        let mut dense_evals = vec![F::ZERO; 8];
        dense_evals[0] = F::from_nonreduced_u32(1);
        dense_evals[5] = F::from_nonreduced_u32(7);
        let dense = MultilinearPoly::new(dense_evals);

        // Test boolean points
        let point1 = vec![F::ZERO, F::ZERO, F::ZERO];
        assert_eq!(sparse.eval_at(&point1), dense.eval_at(&point1));

        let point2 = vec![F::ONE, F::ZERO, F::ONE];
        assert_eq!(sparse.eval_at(&point2), dense.eval_at(&point2));

        let point3 = vec![F::ZERO, F::ONE, F::ZERO];
        assert_eq!(sparse.eval_at(&point3), dense.eval_at(&point3));

        // Test non-boolean points (interpolation)
        let point4 = vec![
            F::from_nonreduced_u32(3),
            F::from_nonreduced_u32(5),
            F::from_nonreduced_u32(2),
        ];
        assert_eq!(sparse.eval_at(&point4), dense.eval_at(&point4));
    }

    #[test]
    fn test_sparse_density() {
        let mut sparse = SparseMultilinearPoly::new(10); // 2^10 = 1024 total

        // Add 10 non-zeros
        for i in 0..10 {
            sparse.insert(i * 100, F::from_nonreduced_u32(i as u32 + 1));
        }

        assert_eq!(sparse.nnz(), 10);
        assert!((sparse.density() - 0.009765625).abs() < 1e-6); // 10/1024
    }

    #[test]
    fn test_sparse_sum_over_hypercube() {
        let mut sparse = SparseMultilinearPoly::new(4);
        sparse.insert(0, F::from_nonreduced_u32(1));
        sparse.insert(3, F::from_nonreduced_u32(5));
        sparse.insert(7, F::from_nonreduced_u32(3));
        sparse.insert(15, F::from_nonreduced_u32(2));

        let sum = sparse.sum_over_hypercube();
        assert_eq!(sum, F::from_nonreduced_u32(11)); // 1 + 5 + 3 + 2
    }

    #[test]
    fn test_sparse_to_dense() {
        let mut sparse = SparseMultilinearPoly::new(3);
        sparse.insert(0, F::from_nonreduced_u32(1));
        sparse.insert(5, F::from_nonreduced_u32(7));

        let dense = sparse.to_dense();
        assert_eq!(dense.len(), 8);
        assert_eq!(dense[0], F::from_nonreduced_u32(1));
        assert_eq!(dense[5], F::from_nonreduced_u32(7));
        assert_eq!(dense[1], F::ZERO);
        assert_eq!(dense[2], F::ZERO);
    }

    #[test]
    fn test_sparse_from_dense() {
        let mut dense_evals = vec![F::ZERO; 8];
        dense_evals[1] = F::from_nonreduced_u32(3);
        dense_evals[4] = F::from_nonreduced_u32(9);

        let sparse = SparseMultilinearPoly::from_dense(dense_evals, 3);
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(1), F::from_nonreduced_u32(3));
        assert_eq!(sparse.get(4), F::from_nonreduced_u32(9));
        assert_eq!(sparse.get(0), F::ZERO);
    }

    #[test]
    fn test_sparse_insert_zero_removes() {
        let mut sparse = SparseMultilinearPoly::new(3);
        sparse.insert(5, F::from_nonreduced_u32(7));
        assert_eq!(sparse.nnz(), 1);

        // Inserting zero should remove the entry
        sparse.insert(5, F::ZERO);
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.get(5), F::ZERO);
    }

    #[test]
    fn test_sparse_lift() {
        let mut sparse = SparseMultilinearPoly::<F>::new(2);
        sparse.insert(0, F::from_nonreduced_u32(2));
        sparse.insert(3, F::from_nonreduced_u32(5));

        let lifted = sparse.lift::<E>();
        assert_eq!(lifted.num_vars(), 2);
        assert_eq!(lifted.nnz(), 2);
        assert_eq!(lifted.get(0), E::from_base(F::from_nonreduced_u32(2)));
        assert_eq!(lifted.get(3), E::from_base(F::from_nonreduced_u32(5)));
    }
}
