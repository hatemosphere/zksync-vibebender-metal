// based on https://github.com/a16z/jolt/blob/main/jolt-core/src/poly/split_eq_poly.rs
use field::Field;
use rayon::prelude::*;

use crate::{EqPoly, MultiLinearPoly, UniPoly};

pub struct SplitEqPoly<F> {
    /// Number of unbound variables remaining (decrements each round).
    pub(crate) current_index: usize,
    /// Accumulated eq(w_bound, r_bound) from already-bound variables.
    pub(crate) current_scalar: F,
    /// The full challenge vector w.
    pub(crate) w: Vec<F>,
    /// Prefix eq tables for w_in. E_in_vec[k] = eq(w_in[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; E_in_vec[0] = [1].
    pub(crate) e_in_vec: Vec<MultiLinearPoly<F>>,
    /// Prefix eq tables for w_out. E_out_vec[k] = eq(w_out[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; E_out_vec[0] = [1].
    pub(crate) e_out_vec: Vec<MultiLinearPoly<F>>,
}

impl<F: Field> SplitEqPoly<F> {
    pub fn new(w: &[F]) -> Self {
        Self::new_with_scaling(w, F::ONE)
    }

    pub fn new_with_scaling(w: &[F], scaling_factor: F) -> Self {
        let m = w.len() / 2;
        let (w_out, w_in) = w.split_last().unwrap().1.split_at(m);

        let (e_out_vec, e_in_vec) =
            rayon::join(|| EqPoly::eval_cached(w_out), || EqPoly::eval_cached(w_in));

        Self {
            current_index: w.len(),
            current_scalar: scaling_factor,
            w: w.to_vec(),
            e_in_vec,
            e_out_vec,
        }
    }

    pub fn bind_in_place(&mut self, r: &F) {
        let mut prod_w_r = self.w[self.current_index - 1];
        prod_w_r.mul_assign(r);
        prod_w_r.double();

        let mut temp = F::ONE;
        temp.sub_assign(&self.w[self.current_index - 1]);
        temp.sub_assign(r);
        temp.add_assign(&prod_w_r);

        self.current_scalar.mul_assign(&temp);

        self.current_index -= 1;

        if self.w.len() / 2 < self.current_index && self.e_in_vec.len() > 1 {
            self.e_in_vec.pop();
        } else if 0 < self.current_index && self.e_out_vec.len() > 1 {
            self.e_out_vec.pop();
        }
    }

    /// Compute the cubic polynomial s(X) = l(X) * q(X), where l(X) is the
    /// current (linear) eq polynomial and q(X) = c + dX + eX^2, given the following:
    /// - q_constant: c, the constant term of q
    /// - q_quadratic: e, the quadratic term of q
    /// - previous_claim: V = s(0) + s(1) from the previous round
    ///
    /// Algorithm:
    /// 1. Get l(X) = aX + b from current eq variable
    /// 2. Solve for d using V = bc + (a+b)(c+d+e)
    /// 3. Expand s(X) = (aX + b)(c + dX + eX²)
    pub fn poly_deg_3(&self, q_constant: F, q_quadratic: F, previous_claim: F) -> UniPoly<F> {
        let c = q_constant;
        let e = q_quadratic;

        // Get l(X) = aX + b from current eq variable
        // eq(w, X) = (1-w)(1-X) + wX = (2w-1)X + (1-w)
        let w = self.w[self.current_index - 1];

        let mut a = w;
        a.double();
        a.sub_assign(&F::ONE);  // a = 2w - 1

        let mut b = F::ONE;
        b.sub_assign(&w);        // b = 1 - w

        // Solve for d: V = bc + (a+b)(c+d+e)
        // => d = (V - bc)/(a+b) - c - e
        let mut bc = b;
        bc.mul_assign(&c);

        let mut numerator = previous_claim;
        numerator.sub_assign(&bc);  // V - bc

        let mut denominator = a;
        denominator.add_assign(&b);  // a + b
        let denom_inv = denominator.inverse().unwrap();

        let mut d = numerator;
        d.mul_assign(&denom_inv);
        d.sub_assign(&c);
        d.sub_assign(&e);

        // Expand s(X) = (aX + b)(c + dX + eX²)
        // = bc + bdX + beX² + acX + adX² + aeX³
        // = aeX³ + (ad + be)X² + (ac + bd)X + bc

        let mut c3 = a;
        c3.mul_assign(&e);  // ae

        let mut c2 = a;
        c2.mul_assign(&d);  // ad
        let mut be = b;
        be.mul_assign(&e);
        c2.add_assign(&be);  // ad + be

        let mut c1 = a;
        c1.mul_assign(&c);  // ac
        let mut bd = b;
        bd.mul_assign(&d);
        c1.add_assign(&bd);  // ac + bd

        let c0 = bc; 

        UniPoly::new(vec![c0, c1, c2, c3])
    }

    /// Compute the quadratic polynomial s(X) = l(X) * q(X), where l(X) is the
    /// current (linear) eq polynomial and q(X) = c + dX, given:
    /// - q_constant: c, the constant term of q
    /// - previous_claim: V = s(0) + s(1) from the previous round
    ///
    /// Algorithm:
    /// 1. Get l(X) = aX + b from current eq variable
    /// 2. Solve for d using V = bc + (a+b)(c+d)
    /// 3. Expand s(X) = (aX + b)(c + dX)
    pub fn poly_deg_2(&self, q_constant: F, previous_claim: F) -> UniPoly<F> {
        let c = q_constant;

        // Get l(X) = aX + b from current eq variable
        let w = self.w[self.current_index - 1];

        let mut a = w;
        a.double();
        a.sub_assign(&F::ONE);  // a = 2w - 1

        let mut b = F::ONE;
        b.sub_assign(&w);        // b = 1 - w

        // Solve for d: V = bc + (a+b)(c+d)
        // => d = (V - bc)/(a+b) - c
        let mut bc = b;
        bc.mul_assign(&c);

        let mut numerator = previous_claim;
        numerator.sub_assign(&bc);  // V - bc

        let mut denominator = a;
        denominator.add_assign(&b);  // a + b
        let denom_inv = denominator.inverse().unwrap();

        let mut d = numerator;
        d.mul_assign(&denom_inv);
        d.sub_assign(&c);

        // Expand s(X) = (aX + b)(c + dX)
        // = bc + bdX + acX + adX²
        // = adX² + (ac + bd)X + bc

        let mut c2 = a;
        c2.mul_assign(&d);  // ad

        let mut c1 = a;
        c1.mul_assign(&c);  // ac
        let mut bd = b;
        bd.mul_assign(&d);
        c1.add_assign(&bd);  // ac + bd

        let c0 = bc;

        UniPoly::new(vec![c0, c1, c2])
    }

    #[inline]
    pub fn par_fold_out_in<
        OuterAcc: Send,
        InnerAcc: Send,
        MakeInner: Fn() -> InnerAcc + Sync + Send,
        InnerStep: Fn(&mut InnerAcc, usize, usize, F) + Sync + Send,
        OuterStep: Fn(usize, F, InnerAcc) -> OuterAcc + Sync + Send,
        Merge: Fn(OuterAcc, OuterAcc) -> OuterAcc + Sync + Send,
    >(
        &self,
        make_inner: MakeInner,
        inner_step: InnerStep,
        outer_step: OuterStep,
        merge: Merge,
    ) -> OuterAcc {
        let e_out = self.e_out_current();
        let e_in = self.e_in_current();
        let out_len = e_out.len();
        let in_len = e_in.len();

        (0..out_len)
            .into_par_iter()
            .map(|x_out| {
                let mut inner_acc = make_inner();

                for x_in in 0..in_len {
                    let g = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, g, x_in, e_in[x_in]);
                }

                outer_step(x_out, e_out[x_out], inner_acc)
            })
            .reduce_with(merge)
            .expect("par_fold_out_in: empty E_out; invariant violation")
    }

    pub fn e_in_current(&self) -> &[F] {
        self.e_in_vec
            .last()
            .expect("e_in_vec is never empty")
            .evals()
    }

    pub fn e_in_current_len(&self) -> usize {
        self.e_in_vec
            .last()
            .expect("E_in_vec is never empty")
            .evals()
            .len()
    }

    pub fn e_out_current(&self) -> &[F] {
        self.e_out_vec
            .last()
            .expect("e_out_vec is never empty")
            .evals()
    }

    fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        let num_x = self.e_in_current_len();

        let num_x_in_bits = if num_x.is_power_of_two() {
            (1usize.leading_zeros() - num_x.leading_zeros()) as usize
        } else {
            (0usize.leading_zeros() - num_x.leading_zeros()) as usize
        };

        (x_out << num_x_in_bits) | x_in
    }
}
