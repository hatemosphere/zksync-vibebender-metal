use field::{Field, FieldExtension};
use poly::{MultiLinearPoly, SplitEqPoly, UniPoly};

use crate::SumcheckInstanceProver;

/// Sumcheck for proving sum_x eq(r, x) * p(x)
pub struct MLEEvalSumcheck<F, E> {
    p_base: Option<MultiLinearPoly<F>>,
    p_ext: Option<MultiLinearPoly<E>>,
    eq: SplitEqPoly<E>,
    num_rounds: usize,
    claim: E,
    current_round: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, E: FieldExtension<F> + Field> MLEEvalSumcheck<F, E> {
    pub fn new(p: MultiLinearPoly<F>, r: &[E]) -> Self {
        assert_eq!(p.num_vars(), r.len());

        let num_rounds = p.num_vars();
        let eq = SplitEqPoly::new(r);

        let claim = todo!();

        Self {
            p_base: Some(p),
            p_ext: None,
            eq,
            num_rounds,
            claim,
            current_round: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Field, E: FieldExtension<F> + Field> SumcheckInstanceProver<F, E> for MLEEvalSumcheck<F, E> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn compute_message(&mut self, _round: usize, previous_claim: &E) -> UniPoly<E> {
        let q_constant = if self.current_round == 0 {
            let p_base = self.p_base.as_ref().unwrap();
            let p_evals = p_base.evals();

            self.eq.par_fold_out_in(
                || E::ZERO,
                |acc: &mut E, g, _x_in, eq_in| {
                    let mut term = eq_in;
                    term.mul_assign_by_base(&p_evals[g]);
                    acc.add_assign(&term);
                },
                |_x_out, eq_out, inner| {
                    let mut c = inner;
                    c.mul_assign(&eq_out);
                    c
                },
                |mut a, b| {
                    a.add_assign(&b);
                    a
                },
            )
        } else {
            let p_ext = self.p_ext.as_ref().unwrap();
            let p_evals = p_ext.evals();

            self.eq.par_fold_out_in(
                || E::ZERO,
                |acc: &mut E, g, _x_in, eq_in| {
                    let p0 = p_evals[g];
                    let mut term = eq_in;
                    term.mul_assign(&p0);
                    acc.add_assign(&term);
                },
                |_x_out, eq_out, inner| {
                    let mut c = inner;
                    c.mul_assign(&eq_out);
                    c
                },
                |mut a, b| {
                    a.add_assign(&b);
                    a
                },
            )
        };

        self.eq.poly_deg_2(q_constant, *previous_claim)
    }

    fn ingest_challenge(&mut self, r: &E, _round: usize) {
        self.eq.bind_in_place(r);

        if self.current_round == 0 {
            let p_base = self.p_base.take().unwrap();
            let p_ext = p_base.bind_first(r);
            self.p_ext = Some(p_ext);
        } else {
            self.p_ext.as_mut().unwrap().bind_in_place(r);
        }

        self.current_round += 1;
    }
}
