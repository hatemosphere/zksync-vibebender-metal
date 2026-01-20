use field::{Field, FieldExtension};
use poly::{MultiLinearPoly, SplitEqPoly, UniPoly};

use crate::SumcheckInstanceProver;

/// Sumcheck for proving sum_x eq(r, x) * p(x) * q(x)
pub struct MLEProductSumcheck<F, E> {
    p_base: Option<MultiLinearPoly<F>>,
    q_base: Option<MultiLinearPoly<F>>,
    p_ext: Option<MultiLinearPoly<E>>,
    q_ext: Option<MultiLinearPoly<E>>,
    eq: SplitEqPoly<E>,
    num_rounds: usize,
    claim: E,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, E: FieldExtension<F> + Field> MLEProductSumcheck<F, E> {
    pub fn new(p: MultiLinearPoly<F>, q: MultiLinearPoly<F>, r: &[E]) -> Self {
        assert_eq!(p.num_vars(), q.num_vars());
        assert_eq!(p.num_vars(), r.len());

        let num_rounds = p.num_vars();
        let eq = SplitEqPoly::new(r);

        let claim = todo!();

        Self {
            p_base: Some(p),
            q_base: Some(q),
            p_ext: None,
            q_ext: None,
            eq,
            num_rounds,
            claim,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Field, E: FieldExtension<F> + Field> SumcheckInstanceProver<F, E> for MLEProductSumcheck<F, E> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn compute_message(&mut self, round: usize, previous_claim: &E) -> UniPoly<E> {
        let (q_constant, q_quadratic) = if round == 0 {

            let p_evals = self.p_base.as_ref().unwrap().evals();
            let q_evals = self.q_base.as_ref().unwrap().evals();
            let half = p_evals.len() / 2;

            self.eq.par_fold_out_in(
                || (E::ZERO, E::ZERO),
                |acc: &mut (E, E), g, _x_in, eq_in| {
                    // TODO: consider indexing polys with gray codes for better memory locality
                    let p0 = &p_evals[g];
                    let p1 = &p_evals[g+half];
                    let q0 = &q_evals[g];
                    let q1 = &q_evals[g+half];

                    // G(0) = p0 * q0
                    let mut term0 = eq_in;
                    term0.mul_assign_by_base(&p0);
                    term0.mul_assign_by_base(&q0);
                    acc.0.add_assign(&term0);

                    // X² coefficient: (p1 - p0) * (q1 - q0)
                    let mut p_diff = *p1;
                    p_diff.sub_assign(&p0);
                    let mut q_diff = *q1;
                    q_diff.sub_assign(&q0);
                    let mut term2 = eq_in;
                    term2.mul_assign_by_base(&p_diff);
                    term2.mul_assign_by_base(&q_diff);
                    acc.1.add_assign(&term2);
                },
                |_x_out, eq_out, inner| {
                    let mut c = inner.0;
                    c.mul_assign(&eq_out);
                    let mut e = inner.1;
                    e.mul_assign(&eq_out);
                    (c, e)
                },
                |mut a, b| {
                    a.0.add_assign(&b.0);
                    a.1.add_assign(&b.1);
                    a
                },
            )
        } else {
            let p_ext = self.p_ext.as_ref().unwrap();
            let q_ext = self.q_ext.as_ref().unwrap();
            let p_evals = p_ext.evals();
            let q_evals = q_ext.evals();
            let half = p_evals.len() / 2;

            self.eq.par_fold_out_in(
                || (E::ZERO, E::ZERO),
                |acc: &mut (E, E), g, _x_in, eq_in| {
                    let p0 = p_evals[g];
                    let p1 = p_evals[g + half];
                    let q0 = q_evals[g];
                    let q1 = q_evals[g + half];

                    let mut term0 = eq_in;
                    term0.mul_assign(&p0);
                    term0.mul_assign(&q0);
                    acc.0.add_assign(&term0);

                    let mut p_diff = p1;
                    p_diff.sub_assign(&p0);
                    let mut q_diff = q1;
                    q_diff.sub_assign(&q0);
                    let mut term2 = eq_in;
                    term2.mul_assign(&p_diff);
                    term2.mul_assign(&q_diff);
                    acc.1.add_assign(&term2);
                },
                |_x_out, eq_out, inner| {
                    let mut c = inner.0;
                    c.mul_assign(&eq_out);
                    let mut e = inner.1;
                    e.mul_assign(&eq_out);
                    (c, e)
                },
                |mut a, b| {
                    a.0.add_assign(&b.0);
                    a.1.add_assign(&b.1);
                    a
                },
            )
        };

        // Use the new poly_deg_3 method to compute s(X) = L(X) * G(X)
        self.eq.poly_deg_3(q_constant, q_quadratic, *previous_claim)
    }

    fn ingest_challenge(&mut self, r: &E, round: usize) {
        self.eq.bind_in_place(r);

        if round == 0 {
            let p_base = self.p_base.take().unwrap();
            let q_base = self.q_base.take().unwrap();

            let p_ext = p_base.bind_first(r);
            let q_ext = q_base.bind_first(r);

            self.p_ext = Some(p_ext);
            self.q_ext = Some(q_ext);
        } else {
            self.p_ext.as_mut().unwrap().bind_in_place(r);
            self.q_ext.as_mut().unwrap().bind_in_place(r);
        }
    }
}
