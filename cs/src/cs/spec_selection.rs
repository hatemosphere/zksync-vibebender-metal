use std::collections::BTreeSet;

use crate::constraint::*;
use crate::cs::circuit_trait::Circuit;
use crate::cs::utils::mask_by_boolean_into_accumulator_constraint;
use crate::cs::utils::mask_linear_term_by_boolean_into_accumulator_constraint;
use crate::cs::Variable;
use crate::types::*;
use crate::witness_placer::*;
use ::field::PrimeField;

#[track_caller]
pub(crate) fn spec_choose_from_orthogonal_variants<F: PrimeField, CS: Circuit<F>>(
    cs: &mut CS,
    flags: &[Boolean],
    variants: &[Num<F>],
    push_next_layer: bool,
    name: &str,
) -> Num<F> {
    assert_eq!(flags.len(), variants.len());

    if flags.iter().all(|el| matches!(el, Boolean::Is(..))) {
        return spec_choose_from_orthogonal_variants_no_nots(
            cs,
            flags,
            variants,
            push_next_layer,
            name,
        );
    }

    // we enumerate all booleans to properly resolve
    let mut boolean_vars = BTreeSet::new();
    for flag in flags.iter() {
        match flag {
            Boolean::Is(var) => {
                boolean_vars.insert(*var);
            }
            Boolean::Not(var) => {
                boolean_vars.insert(*var);
            }
            Boolean::Constant(..) => {
                panic!("constant flags do not make sense for selection functions")
            }
        }
    }

    // now we can make a constraint
    let mut constraint = Constraint::<F>::empty();
    for (flag, variant) in flags.iter().zip(variants.iter()) {
        constraint = mask_by_boolean_into_accumulator_constraint(flag, variant, constraint);
    }

    spec_choose_from_orthogonal_variants_for_constraint_inner(
        cs,
        boolean_vars,
        constraint,
        push_next_layer,
        name,
    )
}

#[track_caller]
pub(crate) fn spec_choose_from_orthogonal_variants_for_linear_terms<
    F: PrimeField,
    CS: Circuit<F>,
>(
    cs: &mut CS,
    flags: &[Boolean],
    variants: &[Constraint<F>],
    push_next_layer: bool,
    name: &str,
) -> Num<F> {
    assert_eq!(flags.len(), variants.len());

    // we enumerate all booleans to properly resolve
    let mut boolean_vars = BTreeSet::new();
    for flag in flags.iter() {
        match flag {
            Boolean::Is(var) => {
                boolean_vars.insert(*var);
            }
            Boolean::Not(var) => {
                boolean_vars.insert(*var);
            }
            Boolean::Constant(..) => {
                panic!("constant flags do not make sense for selection functions")
            }
        }
    }

    // now we can make a constraint
    let mut constraint = Constraint::<F>::empty();
    for (flag, variant) in flags.iter().zip(variants.iter()) {
        constraint =
            mask_linear_term_by_boolean_into_accumulator_constraint(flag, variant, constraint);
    }

    spec_choose_from_orthogonal_variants_for_constraint_inner(
        cs,
        boolean_vars,
        constraint,
        push_next_layer,
        name,
    )
}

fn spec_choose_from_orthogonal_variants_for_constraint_inner<F: PrimeField, CS: Circuit<F>>(
    cs: &mut CS,
    booleans: BTreeSet<Variable>,
    mut constraint: Constraint<F>,
    push_next_layer: bool,
    name: &str,
) -> Num<F> {
    let (quadratic, linear, constant_term) = constraint.clone().split_max_quadratic();

    let mut parsed_quadratic = vec![];
    let mut parsed_linear = vec![];
    for (c, a, b) in quadratic.into_iter() {
        if booleans.contains(&a) {
            assert!(booleans.contains(&b) == false);
            parsed_quadratic.push((a, (c, b)));
        } else {
            assert!(booleans.contains(&b));
            parsed_quadratic.push((b, (c, a)));
        }
    }
    for (c, a) in linear.into_iter() {
        assert!(booleans.contains(&a));
        parsed_linear.push((a, c));
    }

    if push_next_layer {
        let var = cs.add_intermediate_named_variable_from_constraint(constraint, name);
        Num::Var(var)
    } else {
        let result = cs.add_named_variable(name);
        constraint -= Term::from(result);

        // Filter constants equal to 1
        let mut quadratic_trivial = vec![];
        let mut quadratic_nontrivial = vec![];
        for (flag, (c, a)) in parsed_quadratic.into_iter() {
            assert!(c != F::ZERO);
            if c == F::ONE {
                quadratic_trivial.push((flag, a));
            } else {
                quadratic_nontrivial.push((flag, (c, a)));
            }
        }

        let mut linear_trivial = vec![];
        let mut linear_nontrivial = vec![];
        for (flag, c) in parsed_linear.into_iter() {
            assert!(c != F::ZERO);
            if c == F::ONE {
                linear_trivial.push(flag);
            } else {
                linear_nontrivial.push((flag, c));
            }
        }

        let value_fn = move |placer: &mut CS::WitnessPlacer| {
            let mut value =
                <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(constant_term);

            for (mask, term) in quadratic_trivial.iter() {
                let mask = placer.get_boolean(*mask);
                let term = placer.get_field(*term);
                value.add_assign_masked(&mask, &term);
            }

            for (mask, (constant, term)) in quadratic_nontrivial.iter() {
                let mask = placer.get_boolean(*mask);
                let constant = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(*constant);
                let term = placer.get_field(*term);
                value.add_assign_product_masked(&mask, &constant, &term);
            }

            if linear_trivial.len() > 0 {
                let mut mask_for_one_constant =
                    <CS::WitnessPlacer as WitnessTypeSet<F>>::Mask::constant(false);
                for mask in linear_trivial.iter() {
                    let mask = placer.get_boolean(*mask);
                    mask_for_one_constant = mask_for_one_constant.or(&mask);
                }
                let one = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(F::ONE);
                value.add_assign_masked(&mask_for_one_constant, &one);
            }

            for (mask, constant) in linear_nontrivial.iter() {
                let mask = placer.get_boolean(*mask);
                let constant = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(*constant);
                value.add_assign_masked(&mask, &constant);
            }

            placer.assign_field(result, &value);
        };

        cs.set_values(value_fn);

        assert!(constraint.degree() > 0);
        if constraint.degree() == 2 {
            cs.add_constraint(constraint);
        } else {
            cs.add_constraint_allow_explicit_linear(constraint);
        }

        Num::Var(result)
    }
}

#[track_caller]
fn spec_choose_from_orthogonal_variants_no_nots<F: PrimeField, CS: Circuit<F>>(
    cs: &mut CS,
    flags: &[Boolean],
    variants: &[Num<F>],
    push_next_layer: bool,
    name: &str,
) -> Num<F> {
    assert_eq!(flags.len(), variants.len());

    let mut boolean_vars = BTreeSet::new();

    let mut parsed_quadratic = vec![];
    let mut parsed_linear = vec![];

    // now we can make a constraint
    let mut constraint = Constraint::empty();
    for (flag, variant) in flags.iter().zip(variants.iter()) {
        let Boolean::Is(flag) = *flag else {
            unreachable!()
        };

        let is_unique = boolean_vars.insert(flag);
        assert!(is_unique, "use of the same flag in orthogonal combination");

        match variant {
            Num::Var(variant) => {
                constraint = constraint + Term::from(flag) * Term::from(*variant);
                parsed_quadratic.push((flag, *variant));
            }
            Num::Constant(constant) => {
                constraint += Term::from((*constant, flag));
                parsed_linear.push((flag, *constant));
            }
        }
    }

    if push_next_layer {
        let var = cs.add_intermediate_named_variable_from_constraint(constraint, name);
        Num::Var(var)
    } else {
        let result = cs.add_variable();
        constraint -= Term::from(result);

        let value_fn = move |placer: &mut CS::WitnessPlacer| {
            let mut value = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(F::ZERO);

            for (mask, term) in parsed_quadratic.iter() {
                let mask = placer.get_boolean(*mask);
                let term = placer.get_field(*term);
                value.add_assign_masked(&mask, &term);
            }

            for (mask, constant) in parsed_linear.iter() {
                let mask = placer.get_boolean(*mask);
                let constant = <CS::WitnessPlacer as WitnessTypeSet<F>>::Field::constant(*constant);
                value.add_assign_masked(&mask, &constant);
            }

            placer.assign_field(result, &value);
        };

        cs.set_values(value_fn);

        assert!(constraint.degree() > 0);
        if constraint.degree() == 2 {
            cs.add_constraint(constraint);
        } else {
            cs.add_constraint_allow_explicit_linear(constraint);
        }

        Num::Var(result)
    }
}
