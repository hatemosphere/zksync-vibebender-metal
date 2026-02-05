use field::{Field, FieldExtension};

use super::*;

pub fn make_eq_poly_reduced<E: Field>(
    previous_round_challenges: &[E],
) -> Vec<Box<[E]>> {
    assert!(previous_round_challenges.len() > 1);
    // challenges[0] is the challenge used to fold a variable, that is encoded as MSB in the values enumeration,
    // and we will produce the outputs in a same form. We also keep all intermediate forms for simplicity
    let mut result = Vec::with_capacity(previous_round_challenges.len() + 1);
    result.push(vec![E::ONE].into_boxed_slice());

    let mut size = 2;
    let mut idx = previous_round_challenges.len() - 1;
    let f1 = previous_round_challenges[idx];
    let mut f0 = E::ONE;
    f0.sub_assign(&f1);
    let layer = vec![f0, f1].into_boxed_slice();
    result.push(layer);
    for _ in 2..previous_round_challenges.len() {
        size *= 2;
        idx -= 1;
        let mut layer = Box::new_uninit_slice(size);
        let previous_layer = result.last().expect("is present");

        let f1 = previous_round_challenges[idx];
        let mut f0 = E::ONE;
        f0.sub_assign(&f1);

        let half_size = size / 2;

        assert_eq!(previous_layer.len(), half_size);

        for index in 0..half_size {
            let mut left = previous_layer[index];
            let mut right = left;
            left.mul_assign(&f0);
            right.mul_assign(&f1);
            layer[index].write(left);
            layer[index + half_size].write(right);
        }

        result.push(unsafe { layer.assume_init() });
    }

    result
}

pub fn make_eq_poly_in_full<E: Field>(
    previous_round_challenges: &[E],
) -> Vec<Box<[E]>> {
    let mut result = make_eq_poly_reduced(previous_round_challenges);
    assert_eq!(result.len(), previous_round_challenges.len());

    {
        let previous_layer = result.last().expect("is present");
        let size = previous_layer.len() * 2;
        let mut layer = Box::new_uninit_slice(size);

        let f1 = previous_round_challenges[0];
        let mut f0 = E::ONE;
        f0.sub_assign(&f1);

        let half_size = size / 2;

        assert_eq!(previous_layer.len(), half_size);

        for index in 0..half_size {
            let mut left = previous_layer[index];
            let mut right = left;
            left.mul_assign(&f0);
            right.mul_assign(&f1);
            layer[index].write(left);
            layer[index + half_size].write(right);
        }
        result.push(unsafe { layer.assume_init() });
    }

    result
}

pub(crate) fn evaluate_with_precomputed_eq<F: PrimeField, E: FieldExtension<F> + Field>(
    base_field_values: &[F],
    eq: &[E],
) -> E {
    assert_eq!(base_field_values.len(), eq.len());
    let mut result = E::ZERO;
    for (a, b) in base_field_values.iter().zip(eq.iter()) {
        let mut t = *b;
        t.mul_assign_by_base(a);
        result.add_assign(&t);
    }

    result
}

pub(crate) fn evaluate_with_precomputed_eq_ext<E: Field>(
    ext_field_values: &[E],
    eq: &[E],
) -> E {
    assert_eq!(ext_field_values.len(), eq.len());
    let mut result = E::ZERO;
    for (a, b) in ext_field_values.iter().zip(eq.iter()) {
        let mut t = *b;
        t.mul_assign(a);
        result.add_assign(&t);
    }

    result
}

pub(crate) fn evaluate_constant_and_quadratic_coeffs_with_precomputed_eq<
    F: PrimeField,
    E: FieldExtension<F> + Field,
>(
    values: &[[E; 2]],
    eq: &[E],
) -> [E; 2] {
    assert_eq!(values.len(), eq.len());
    let mut result_0 = E::ZERO;
    let mut result_1 = E::ZERO;
    for (a, b) in values.iter().zip(eq.iter()) {
        let [a0, a1] = a;

        let mut t0 = *b;
        t0.mul_assign(a0);
        result_0.add_assign(&t0);

        let mut t1 = *b;
        t1.mul_assign(a1);
        result_1.add_assign(&t1);
    }

    [result_0, result_1]
}
