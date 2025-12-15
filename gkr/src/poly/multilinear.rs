use super::{eq_poly::EqPoly, Group, Polynomial, Set, SumcheckPoly};
use crate::PAR_THRESHOLD;
use core::ops::{Add, Neg, Sub};
use field::{Field, FieldExtension};
use rayon::prelude::*;
use std::{cmp::Ordering, ops::Mul};

/// Multilinear polynomial represented by evaluations on the boolean hypercube {0,1}^n
/// We assume it is zero padded to the next power of two
#[derive(Clone, Debug)]
pub struct MultilinearPoly<T> {
    evals: Vec<T>,
    num_vars: usize,
}

impl<F: Field> MultilinearPoly<F> {
    pub fn new(evals: Vec<F>) -> Self {
        let num_vars = evals.len().next_power_of_two().ilog2() as usize;
        Self { evals, num_vars }
    }

    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    pub(crate) fn eval_split_eq_serial(&self, eq_one: &[F], eq_two: &[F]) -> Self {
        let n = (1 << self.num_vars) / (eq_one.len() * eq_two.len());
        let evals = (0..n)
            .map(|i| {
                (0..eq_one.len())
                    .map(|x1| {
                        let partial_sum = (0..eq_two.len())
                            .map(|x2| {
                                let j = x1 * (eq_two.len() * n) + x2 * n + i;
                                let mut tmp = eq_two[x2];
                                tmp.mul_assign(self.evals.get(j).unwrap_or(&F::ZERO));
                                tmp
                            })
                            .fold(F::ZERO, |mut acc, val| {
                                acc.add_assign(&val);
                                acc
                            });
                        let mut tmp = eq_one[x1];
                        tmp.mul_assign(&partial_sum);
                        tmp
                    })
                    .fold(F::ZERO, |mut acc, val| {
                        acc.add_assign(&val);
                        acc
                    })
            })
            .collect();

        Self::new(evals)
    }

    pub(crate) fn eval_split_eq_par(&self, eq_one: &[F], eq_two: &[F]) -> Self {
        let n = (1 << self.num_vars()) / (eq_one.len() * eq_two.len());
        let evals = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..eq_one.len())
                    .into_iter()
                    .map(|x1| {
                        let partial_sum = (0..eq_two.len())
                            .into_iter()
                            .map(|x2| {
                                let j = x1 * (eq_two.len() * n) + x2 * n + i;
                                let mut tmp = eq_two[x2];
                                tmp.mul_assign(self.evals.get(j).unwrap_or(&F::ZERO));
                                tmp
                            })
                            .fold(F::ZERO, |mut acc, val| {
                                acc.add_assign(&val);
                                acc
                            });
                        let mut tmp = eq_one[x1];
                        tmp.mul_assign(&partial_sum);
                        tmp
                    })
                    .fold(F::ZERO, |mut acc, val| {
                        acc.add_assign(&val);
                        acc
                    })
            })
            .collect();

        Self::new(evals)
    }

    pub fn tensor(&self, other: &Self) -> Self {
        let len = 1 << (self.num_vars + other.num_vars);

        if self.is_zero() || other.is_zero() {
            return Self::new(vec![F::ZERO; len]);
        }

        let other_len = 1 << other.num_vars;
        let mut evals = vec![F::ZERO; len];

        if len < PAR_THRESHOLD {
            for (i, &self_eval) in self.evals.iter().enumerate() {
                let chunk_start = i * other_len;
                for (j, &other_eval) in other.evals.iter().enumerate() {
                    let mut val = self_eval;
                    val.mul_assign(&other_eval);
                    evals[chunk_start + j] = val;
                }
            }
        } else {
            evals
                .par_chunks_mut(other_len)
                .take(self.evals.len())
                .enumerate()
                .for_each(|(i, chunk)| {
                    let self_eval = self.evals[i];
                    for (j, &other_eval) in other.evals.iter().enumerate() {
                        let mut val = self_eval;
                        val.mul_assign(&other_eval);
                        chunk[j] = val;
                    }
                });
        }

        Self::new(evals)
    }
}

impl<F: Field> Default for MultilinearPoly<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: Field> Polynomial<F> for MultilinearPoly<F> {
    fn eval_at(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), self.num_vars());

        let evals = self.partial_eval(r).evals;

        debug_assert_eq!(evals.len(), 1);
        evals[0]
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self, _index: usize) -> usize {
        1
    }

    fn sum_over_hypercube(&self) -> F {
        let sum = self.partial_sum(self.num_vars());
        debug_assert_eq!(sum.evals.len(), 1);
        sum.evals[0]
    }

    fn lift<E: FieldExtension<F> + Field>(self) -> MultilinearPoly<E> {
        MultilinearPoly::<E>::new(self.evals.into_iter().map(E::from_base).collect())
    }
}

impl<F: Field> SumcheckPoly<F> for MultilinearPoly<F> {
    fn partial_eval(&self, r: &[F]) -> Self {
        let mid = r.len() / 2;
        let (r2, r1) = r.split_at(mid);
        let (eq_one, eq_two) = rayon::join(|| EqPoly::evals(r2), || EqPoly::evals(r1));

        if self.evals.len() < PAR_THRESHOLD {
            self.eval_split_eq_serial(&eq_one, &eq_two)
        } else {
            self.eval_split_eq_par(&eq_one, &eq_two)
        }
    }

    fn partial_sum(&self, k: usize) -> Self {
        (0..1 << k)
            .into_par_iter()
            .map(|i| {
                (0..k)
                    .rev()
                    .map(|j| if (i >> j) & 1 == 1 { F::ONE } else { F::ZERO })
                    .collect::<Vec<F>>()
            })
            .map(|x| self.partial_eval(&x))
            .reduce(|| Self::zero(), |acc, val| acc + val)
    }
}

impl<F: Field> PartialEq for MultilinearPoly<F> {
    fn eq(&self, other: &Self) -> bool {
        let self_len = self.evals.len();
        let other_len = other.evals.len();
        match self_len.cmp(&other_len) {
            Ordering::Equal => self.evals == other.evals,
            Ordering::Less => {
                self.evals == other.evals[..self_len]
                    && other.evals[self_len..].iter().all(|x| x.is_zero())
            }
            Ordering::Greater => {
                self.evals[..other_len] == other.evals
                    && self.evals[other_len..].iter().all(|x| x.is_zero())
            }
        }
    }
}

impl<F: Field> Eq for MultilinearPoly<F> {}

impl<F: Field> Add for MultilinearPoly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<F: Field> Add<&Self> for MultilinearPoly<F> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let (short, mut long) = if self.evals.len() < rhs.evals.len() {
            (&self.evals, rhs.evals.clone())
        } else {
            (&rhs.evals, self.evals)
        };

        if long.len() < PAR_THRESHOLD {
            long.iter_mut().zip(short.iter()).for_each(|(x, y)| {
                x.add_assign(y);
            });
        } else {
            long.par_iter_mut()
                .zip(short.par_iter())
                .for_each(|(x, y)| {
                    x.add_assign(y);
                });
        }
        Self::new(long)
    }
}

impl<F: Field> Neg for MultilinearPoly<F> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        if self.evals.len() < PAR_THRESHOLD {
            self.evals.iter_mut().for_each(|x| {
                x.negate();
            });
        } else {
            self.evals.par_iter_mut().for_each(|x| {
                x.negate();
            })
        }

        self
    }
}

impl<F: Field> Sub for MultilinearPoly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<F: Field> Sub<&Self> for MultilinearPoly<F> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        let (short, mut long) = if self.evals.len() < rhs.evals.len() {
            (&self.evals, rhs.evals.clone())
        } else {
            (&rhs.evals, self.evals)
        };

        if long.len() < PAR_THRESHOLD {
            long.iter_mut().zip(short.iter()).for_each(|(x, y)| {
                x.sub_assign(y);
            });
        } else {
            long.par_iter_mut()
                .zip(short.par_iter())
                .for_each(|(x, y)| {
                    x.sub_assign(y);
                });
        }
        Self::new(long)
    }
}

impl<F: Field> Set for MultilinearPoly<F> {}

impl<F: Field> Group for MultilinearPoly<F> {
    fn zero() -> Self {
        Self::new(vec![])
    }

    fn is_zero(&self) -> bool {
        self.evals.iter().all(|x| x.is_zero())
    }
}

impl<F: Field> Mul<F> for MultilinearPoly<F> {
    type Output = Self;

    fn mul(self, s: F) -> Self::Output {
        self * &s
    }
}

impl<'a, F: Field> Mul<&'a F> for MultilinearPoly<F> {
    type Output = Self;

    fn mul(self, s: &F) -> Self::Output {
        Self::new(
            self.evals
                .into_iter()
                .map(|mut x| {
                    x.mul_assign(s);
                    x
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::test_group_prop;
    use field::{Mersenne31Field, PrimeField};
    use proptest::{
        arbitrary::{any, Arbitrary},
        collection::vec,
        prelude::{BoxedStrategy, Strategy},
        prop_assert, prop_assert_eq, proptest,
    };

    type F = Mersenne31Field;
    type P = MultilinearPoly<F>;

    impl Arbitrary for P {
        type Parameters = ();

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            proptest::collection::vec(any::<u32>(), 1..=64)
                .prop_map(|evals| P::new(evals.into_iter().map(F::from_nonreduced_u32).collect()))
                .boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    #[test]
    fn group_props() {
        test_group_prop::<P>();
    }

    #[test]
    fn test_num_vars() {
        proptest!(|(f: P)| {
            prop_assert_eq!(f.evals.len().next_power_of_two(), 1 << f.num_vars);
        })
    }

    /// MLE should match original evaluations at boolean points
    #[test]
    fn mle_matches_hypercube_evals() {
        proptest!(|(evals: Vec<u32>)| {
            let n = evals.len().next_power_of_two().ilog2() as usize;
            let mut evals: Vec<F> = evals.into_iter().map(F::from_nonreduced_u32).collect();
            evals.resize(1 << n, F::ZERO);

            let poly = MultilinearPoly::new(evals.clone());

            for (i, val) in evals.into_iter().enumerate() {
                let point: Vec<F> = (0..n)
                    .rev()
                    .map(|j| F::from_boolean((i >> j) & 1 == 1))
                    .collect();
                prop_assert_eq!(poly.eval_at(&point), val, "point: {:?}, index: {:?}", point, i);
            }
        })
    }

    #[test]
    fn eval_split_eq_serial_par_consistency() {
        proptest!(|(evals in vec(0..u32::MAX, 1..256))| {
            let n = evals.len().next_power_of_two().ilog2() as usize;
            let mut padded: Vec<F> = evals.into_iter().map(F::from_nonreduced_u32).collect();
            padded.resize(1 << n, F::ZERO);

            let poly = MultilinearPoly::new(padded);
            let mid = n / 2;

            // Generate random eq evals
            let eq_one = EqPoly::<F>::evals(
                &(0..mid).map(|i| F::from_nonreduced_u32(i as u32 * 17)).collect::<Vec<_>>()
            );
            let eq_two = EqPoly::<F>::evals(
                &(0..(n - mid)).map(|i| F::from_nonreduced_u32(i as u32 * 31)).collect::<Vec<_>>()
            );

            let serial = poly.eval_split_eq_serial(&eq_one, &eq_two);
            let par = poly.eval_split_eq_par(&eq_one, &eq_two);
            prop_assert_eq!(serial, par);
        })
    }

    #[test]
    fn test_tensor_prop() {
        proptest!(|(f: P, g: P, h: P)| {
            let one = P::new(vec![F::ONE]);

            prop_assert!(f.tensor(&P::zero()).is_zero());
            prop_assert_eq!(f.clone(), f.tensor(&one));

            let fg = f.tensor(&g);
            let gf = g.tensor(&f);

            if fg.is_zero() {
                prop_assert!(f.is_zero() || g.is_zero());
            } else {
                prop_assert_eq!(
                        fg.evals.len().next_power_of_two(),
                        f.evals.len().next_power_of_two() * g.evals.len().next_power_of_two()
                );

                prop_assert_eq!(fg.num_vars, f.num_vars + g.num_vars);
            }

            // comutative up to isomorphism
            let mut fg = fg.evals;
            let mut gf = gf.evals;
            fg.sort();
            gf.sort();
            prop_assert_eq!(fg, gf);

            // associativity
            prop_assert_eq!(f.tensor(&g).tensor(&h), f.tensor(&g.tensor(&h)));

            // bilinearity
            // prop_assert_eq!(f.tensor(&(g.clone() + h.clone())), f.tensor(&g) + f.tensor(&h));
        })
    }
}
