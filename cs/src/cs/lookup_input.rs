use super::*;
use crate::constraint::Constraint;
use crate::witness_placer::*;
use field::PrimeField;

impl<F: PrimeField> LookupInput<F> {
    pub fn empty() -> Self {
        LookupInput::Expression {
            linear_terms: vec![],
            constant_coeff: F::ZERO,
        }
    }

    pub fn evaluate<W: WitnessPlacer<F>>(&self, placer: &mut W) -> W::Field {
        match self {
            LookupInput::Variable(variable) => placer.get_field(*variable),
            LookupInput::Expression {
                linear_terms,
                constant_coeff,
            } => {
                let mut result: W::Field = WitnessComputationalField::constant(*constant_coeff);
                for (c, a) in linear_terms.iter() {
                    result.add_assign_product(
                        &WitnessComputationalField::constant(*c),
                        &placer.get_field(*a),
                    );
                }

                result
            }
        }
    }
}

impl<F: PrimeField> From<F> for LookupInput<F> {
    fn from(value: F) -> Self {
        Self::Expression {
            linear_terms: vec![],
            constant_coeff: value,
        }
    }
}

impl<F: PrimeField> From<Variable> for LookupInput<F> {
    fn from(value: Variable) -> Self {
        Self::Variable(value)
    }
}

impl<F: PrimeField> From<Constraint<F>> for LookupInput<F> {
    #[track_caller]
    fn from(value: Constraint<F>) -> Self {
        // NOTE: we allow literal constants here
        assert!(value.degree() <= 1);
        let mut value = value;
        value.normalize();
        let (_, linear_terms, constant_coeff) = value.split_max_quadratic();
        Self::Expression {
            linear_terms,
            constant_coeff,
        }
    }
}
