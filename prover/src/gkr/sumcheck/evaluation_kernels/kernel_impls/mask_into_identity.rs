use super::*;

pub struct MaskIntoIdentityProductGKRRelation {
    pub input: GKRAddress,
    pub mask: GKRAddress,
    pub output: GKRAddress,
}

impl<F: PrimeField, E: FieldExtension<F> + Field> BatchedGKRKernel<F, E>
    for MaskIntoIdentityProductGKRRelation
{
    fn num_challenges(&self) -> usize {
        1
    }

    fn get_inputs(&self) -> GKRInputs {
        GKRInputs {
            inputs_in_base: Vec::new(),
            inputs_in_extension: vec![self.input, self.mask],
            outputs_in_base: Vec::new(),
            outputs_in_extension: vec![self.output],
        }
    }

    fn evaluate_forward_over_storage(
        &self,
        storage: &mut GKRStorage<F, E>,
        expected_output_layer: usize,
        trace_len: usize,
        worker: &Worker,
    ) {
        let kernel = MaskIntoIdentityProductGKRRelationKernel::default();
        let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);
        forward_evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
            &kernel,
            &inputs,
            storage,
            expected_output_layer,
            trace_len,
            worker,
        );
    }

    fn evaluate_over_storage(
        &self,
        storage: &mut GKRStorage<F, E>,
        step: usize,
        batch_challenges: &[E],
        folding_challenges: &[E],
        accumulator: &mut [[E; 2]],
        total_sumcheck_rounds: usize,
        last_evaluations: &mut BTreeMap<GKRAddress, [E; 2]>,
        worker: &Worker,
    ) {
        assert_eq!(
            batch_challenges.len(),
            <Self as BatchedGKRKernel<F, E>>::num_challenges(self)
        );
        let kernel = MaskIntoIdentityProductGKRRelationKernel::default();
        let inputs = <Self as BatchedGKRKernel<F, E>>::get_inputs(self);

        evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs(
            &kernel,
            &inputs,
            storage,
            step,
            batch_challenges,
            folding_challenges,
            accumulator,
            total_sumcheck_rounds,
            last_evaluations,
            worker,
        );
    }
}

#[derive(Default)]
pub struct MaskIntoIdentityProductGKRRelationKernel<F: PrimeField, E: FieldExtension<F> + Field> {
    _marker: core::marker::PhantomData<(F, E)>,
}

impl<F: PrimeField, E: FieldExtension<F> + Field>
    ExtensionFieldInOutFixedSizesEvaluationKernel<F, E, 2, 1>
    for MaskIntoIdentityProductGKRRelationKernel<F, E>
{
    // The quadratic coefficient is La * Lm, NOT kernel([La, Lm]) which would include (1 - Lm).
    fn evaluate_first_round<
        S: super::EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
        SOUT: super::EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
    >(
        &self,
        index: usize,
        sources: &[S; 2],
        output_sources: &[SOUT; 1],
        batch_challenges: &[E; 1],
    ) -> [E; 2] {
        // constant term comes from output.f0 (like the default implementation)
        let output_f0 = output_sources[0].get_f0_only(index).into_value();
        let mut eval_c0 = batch_challenges[0];
        eval_c0.mul_assign(&output_f0);

        // quadratic coefficient = La * Lm (product of slopes)
        let input_slope = sources[0].get_f1_minus_f0_only(index).into_value();
        let mask_slope = sources[1].get_f1_minus_f0_only(index).into_value();

        let mut c2 = input_slope;
        c2.mul_assign(&mask_slope);

        let mut eval_c2 = batch_challenges[0];
        eval_c2.mul_assign(&c2);

        [eval_c0, eval_c2]
    }

    // Override evaluate because this kernel has an affine term (1 - mask).
    // The sumcheck polynomial s(X) = input(X) * mask(X) + (1 - mask(X))
    // Let input(X) = a0 + La*X, mask(X) = m0 + Lm*X
    // Then s(X) = (a0*m0 + 1 - m0) + (a0*Lm + La*m0 - Lm)*X + La*Lm*X²
    //
    // The quadratic coefficient is La*Lm, NOT kernel([La, Lm]) which would incorrectly
    // include the (1 - Lm) affine term.
    fn evaluate<
        S: super::EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
        const EXPLICIT_FORM: bool,
    >(
        &self,
        index: usize,
        sources: &[S; 2],
        batch_challenges: &[E; 1],
    ) -> [E; 2] {
        if EXPLICIT_FORM {
            // For explicit form (final round), return [kernel(f0), kernel(f1)]
            let [input_f0, input_f1] = sources[0].get_two_points::<true>(index);
            let [mask_f0, mask_f1] = sources[1].get_two_points::<true>(index);

            let input_f0_val = input_f0.into_value();
            let mask_f0_val = mask_f0.into_value();
            let input_f1_val = input_f1.into_value();
            let mask_f1_val = mask_f1.into_value();

            // kernel(a, m) = a * m + (1 - m)
            let mut k0 = input_f0_val;
            k0.mul_assign(&mask_f0_val);
            let mut one_minus_m0 = E::ONE;
            one_minus_m0.sub_assign(&mask_f0_val);
            k0.add_assign(&one_minus_m0);

            let mut k1 = input_f1_val;
            k1.mul_assign(&mask_f1_val);
            let mut one_minus_m1 = E::ONE;
            one_minus_m1.sub_assign(&mask_f1_val);
            k1.add_assign(&one_minus_m1);

            let mut eval_c0 = batch_challenges[0];
            eval_c0.mul_assign(&k0);
            let mut eval_c1 = batch_challenges[0];
            eval_c1.mul_assign(&k1);
            [eval_c0, eval_c1]
        } else {
            // For non-explicit form, return [constant_term, quadratic_coeff]
            // constant_term = kernel(input.f0, mask.f0) = input.f0 * mask.f0 + (1 - mask.f0)
            // quadratic_coeff = La * Lm = (input.f1 - input.f0) * (mask.f1 - mask.f0)
            let [input_f0, input_slope] = sources[0].get_two_points::<false>(index);
            let [mask_f0, mask_slope] = sources[1].get_two_points::<false>(index);

            let input_f0_val = input_f0.into_value();
            let mask_f0_val = mask_f0.into_value();
            let input_slope_val = input_slope.into_value();
            let mask_slope_val = mask_slope.into_value();

            // constant_term = input.f0 * mask.f0 + (1 - mask.f0)
            let mut c0 = input_f0_val;
            c0.mul_assign(&mask_f0_val);
            let mut one_minus_m0 = E::ONE;
            one_minus_m0.sub_assign(&mask_f0_val);
            c0.add_assign(&one_minus_m0);

            // quadratic_coeff = La * Lm (just the product of slopes, no affine term!)
            let mut c2 = input_slope_val;
            c2.mul_assign(&mask_slope_val);

            let mut eval_c0 = batch_challenges[0];
            eval_c0.mul_assign(&c0);
            let mut eval_c2 = batch_challenges[0];
            eval_c2.mul_assign(&c2);
            [eval_c0, eval_c2]
        }
    }

    #[inline(always)]
    fn pointwise_eval(&self, input: &[ExtensionFieldRepresentation<F, E>; 2]) -> [E; 1] {
        let [val, mask] = input;
        let val = val.into_value();
        let mask = mask.into_value();
        // input * mask + (1 - mask)
        let mut result = val;
        result.mul_assign(&mask);
        let mut one_minus_mask = E::ONE;
        one_minus_mask.sub_assign(&mask);
        result.add_assign(&one_minus_mask);
        [result]
    }
}