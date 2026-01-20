use field::{Field, FieldExtension};
use rayon::prelude::*;

use crate::PAR_THRESHOLD;


#[derive(Clone, Debug)]
pub struct MultiLinearPoly<T> {
    evals: Vec<T>,
    num_vars: usize,
}

impl<F: Field> MultiLinearPoly<F> {
    pub fn from_evals(evals: Vec<F>) -> Self {
        assert!(evals.len().is_power_of_two());
        let num_vars = evals.len().ilog2() as usize;
        Self { evals, num_vars }
    }

    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn bind_first<E: FieldExtension<F> + Field>(self, r: &E) -> MultiLinearPoly<E> {
        let n = self.evals.len() / 2;

        let (low, high) = self.evals.split_at(n);

        let new_evals = if n < PAR_THRESHOLD {
            low.iter()
                .zip(high)
                .map(|(l, h)| {
                    let mut diff = *h;
                    diff.sub_assign(l);

                    let mut r = *r;
                    r.mul_assign_by_base(&diff);
                    r.add_assign_base(l);
                    r
                })
                .collect()
        } else {
            low.par_iter()
                .zip(high)
                .map(|(l, h)| {
                    let mut diff = *h;
                    diff.sub_assign(l);

                    let mut r = *r;
                    r.mul_assign_by_base(&diff);
                    r.add_assign_base(l);
                    r
                })
                .collect()
        };

        MultiLinearPoly::from_evals(new_evals)
    }

    pub fn bind_in_place(&mut self, r: &F) {
        let n = self.evals.len() / 2;

        let (low, high) = self.evals.split_at_mut(n);

        if n < PAR_THRESHOLD {
            low.iter_mut().zip(high).for_each(|(l, h)| {
                h.sub_assign(l);
                h.mul_assign(r);
                l.add_assign(h);
            });
        } else {
            low.par_iter_mut().zip(high).for_each(|(l, h)| {
                h.sub_assign(l);
                h.mul_assign(r);
                l.add_assign(h);
            });
        }
        self.evals.truncate(n);
    }
}
