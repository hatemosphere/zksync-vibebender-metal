use super::*;
use std::mem::MaybeUninit;

pub trait ExtensionFieldInOutFixedSizesEvaluationKernel<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const IN: usize,
    const OUT: usize,
>
{
    fn evaluate_first_round<
        S: EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
        SOUT: EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
    >(
        &self,
        index: usize,
        sources: &[S; IN],
        output_sources: &[SOUT; OUT],
        batch_challenges: &[E; OUT],
    ) -> [E; 2] {
        assert!(IN > 0);
        assert!(OUT > 0);
        unsafe {
            let mut result = [const { MaybeUninit::uninit() }; 2];
            // we have access to output
            {
                let outputs = output_sources.each_ref().map(|el| el.get_f0_only(index));
                let mut eval = batch_challenges[0];
                eval.mul_assign(&outputs[0].into_value());
                for i in 0..OUT {
                    let mut t = batch_challenges[i];
                    t.mul_assign(&outputs[i].into_value());
                    eval.add_assign(&t);
                }
                result[0].write(eval);
            }
            {
                let sources = sources.each_ref().map(|el| el.get_f1_minus_f0_only(index));
                let evals = self.pointwise_eval(&sources);
                let mut eval = batch_challenges[0];
                eval.mul_assign(&evals[0]);
                for i in 0..OUT {
                    let mut t = batch_challenges[i];
                    t.mul_assign(&evals[i]);
                    eval.add_assign(&t);
                }
                result[1].write(eval);
            }

            result.map(|el| el.assume_init())
        }
    }

    fn evaluate<
        S: EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
        const EXPLICIT_FORM: bool,
    >(
        &self,
        index: usize,
        sources: &[S; IN],
        batch_challenges: &[E; OUT],
    ) -> [E; 2] {
        assert!(IN > 0);
        assert!(OUT > 0);
        unsafe {
            let mut result = [const { MaybeUninit::uninit() }; 2];
            let mut p0s = [const { MaybeUninit::uninit() }; IN];
            let mut p1s = [const { MaybeUninit::uninit() }; IN];
            for i in 0..IN {
                let [f0, f1] = sources[i].get_two_points::<EXPLICIT_FORM>(index);
                p0s[i].write(f0);
                p1s[i].write(f1);
            }
            let p0s = p0s.map(|el| el.assume_init());
            let p1s = p1s.map(|el| el.assume_init());

            for (j, p) in [&p0s, &p1s].into_iter().enumerate() {
                let evals = self.pointwise_eval(p);
                let mut eval = batch_challenges[0];
                eval.mul_assign(&evals[0]);
                for i in 0..OUT {
                    let mut t = batch_challenges[i];
                    t.mul_assign(&evals[i]);
                    eval.add_assign(&t);
                }
                result[j].write(eval);
            }

            result.map(|el| el.assume_init())
        }
    }

    fn pointwise_eval(&self, input: &[ExtensionFieldRepresentation<F, E>; IN]) -> [E; OUT];

    #[inline(always)]
    fn pointwise_eval_forward(&self, input: &[ExtensionFieldRepresentation<F, E>; IN]) -> [E; OUT] {
        self.pointwise_eval(input)
    }
}

fn evaluate_extension_field_in_out_fixed_sizes_evaluation_kernel<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const IN: usize,
    const OUT: usize,
    K: ExtensionFieldInOutFixedSizesEvaluationKernel<F, E, IN, OUT>,
    S: EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
    const EXPLICIT_FORM: bool,
>(
    kernel: &K,
    index: usize,
    sources: &[S],
    batch_challenges: &[E],
) -> [E; 2] {
    debug_assert_eq!(sources.len(), IN);
    debug_assert_eq!(batch_challenges.len(), OUT);
    unsafe {
        let inputs = sources.as_chunks::<IN>().0.iter().next().unwrap_unchecked();
        let challenges = batch_challenges
            .as_chunks::<OUT>()
            .0
            .iter()
            .next()
            .unwrap_unchecked();
        K::evaluate::<S, EXPLICIT_FORM>(kernel, index, inputs, challenges)
    }
}

fn evaluate_extension_field_in_out_fixed_sizes_evaluation_kernel_first_round<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const IN: usize,
    const OUT: usize,
    K: ExtensionFieldInOutFixedSizesEvaluationKernel<F, E, IN, OUT>,
    S: EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
    SOUT: EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>,
>(
    kernel: &K,
    index: usize,
    sources: &[S],
    outputs: &[SOUT],
    batch_challenges: &[E],
) -> [E; 2] {
    debug_assert_eq!(sources.len(), IN);
    debug_assert_eq!(outputs.len(), OUT);
    debug_assert_eq!(batch_challenges.len(), OUT);
    unsafe {
        let inputs = sources.as_chunks::<IN>().0.iter().next().unwrap_unchecked();
        let outputs = outputs
            .as_chunks::<OUT>()
            .0
            .iter()
            .next()
            .unwrap_unchecked();
        let challenges = batch_challenges
            .as_chunks::<OUT>()
            .0
            .iter()
            .next()
            .unwrap_unchecked();
        K::evaluate_first_round::<S, SOUT>(kernel, index, inputs, outputs, challenges)
    }
}

pub fn evaluate_single_input_type_fixed_in_out_kernel_with_extension_inputs<
    F: PrimeField,
    E: FieldExtension<F> + Field,
    const IN: usize,
    const OUT: usize,
    K: ExtensionFieldInOutFixedSizesEvaluationKernel<F, E, IN, OUT>,
>(
    kernel: &K,
    inputs: &GKRInputs,
    storage: &mut GKRStorage<F, E>,
    step: usize,
    batch_challenges: &[E],
    folding_challenges: &[E],
    accumulator: &mut [[E; 2]],
    total_sumcheck_rounds: usize,
    last_evaluations: &mut BTreeMap<GKRAddress, [E; 2]>,
) {
    unsafe {
        // parallelize eventually
        match step {
            0 => {
                let sources = storage.select_for_first_round(inputs);
                assert!(sources.base_field_inputs.is_empty());
                assert!(sources.base_field_outputs.is_empty());
                if sources.extension_field_outputs.is_empty() == false {
                    assert_eq!(sources.extension_field_inputs.len(), IN);
                    assert_eq!(sources.extension_field_outputs.len(), OUT);
                    assert_eq!(batch_challenges.len(), OUT);

                    let inputs = sources
                        .extension_field_inputs
                        .as_chunks::<IN>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();
                    let outputs = sources
                        .extension_field_outputs
                        .as_chunks::<OUT>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();
                    let challenges = batch_challenges
                        .as_chunks::<OUT>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();

                    for index in 0..accumulator.len() {
                        let value = kernel.evaluate_first_round(index, inputs, outputs, challenges);
                        for i in 0..2 {
                            accumulator[index][i].add_assign(&value[i]);
                        }
                    }
                } else {
                    assert_eq!(sources.extension_field_inputs.len(), IN);
                    assert_eq!(batch_challenges.len(), OUT);

                    let inputs = sources
                        .extension_field_inputs
                        .as_chunks::<IN>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();
                    let challenges = batch_challenges
                        .as_chunks::<OUT>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();

                    for index in 0..accumulator.len() {
                        let value = kernel.evaluate::<_, false>(index, inputs, challenges);
                        for i in 0..2 {
                            accumulator[index][i].add_assign(&value[i]);
                        }
                    }
                }
                // for input in sources.extension_field_inputs.iter() {
                //     dbg!(input.current_values());
                // }
                // for output in sources.extension_field_outputs.iter() {
                //     dbg!(output.current_values());
                // }
            }
            i if i + 1 == total_sumcheck_rounds => {
                let sources = storage.select_for_second_round(inputs, folding_challenges);

                {
                    assert!(sources.base_field_inputs.is_empty());
                    assert_eq!(sources.extension_field_inputs.len(), IN);
                    assert_eq!(batch_challenges.len(), OUT);

                    let inputs = sources
                        .extension_field_inputs
                        .as_chunks::<IN>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();
                    let challenges = batch_challenges
                        .as_chunks::<OUT>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();

                    for index in 0..accumulator.len() {
                        let value = kernel.evaluate::<_, true>(index, inputs, challenges);
                        for i in 0..2 {
                            accumulator[index][i].add_assign(&value[i]);
                        }
                    }
                }

                println!("COLLECTING LAST LAYER VALUES");

                for source in sources.extension_field_inputs.iter() {
                    dbg!(source.current_values());
                }

                // Fill the storage
                sources.collect_last_values(inputs, last_evaluations);
            }
            1.. => {
                let sources = storage.select_for_second_round(inputs, folding_challenges);
                {
                    assert!(sources.base_field_inputs.is_empty());
                    assert_eq!(sources.extension_field_inputs.len(), IN);
                    assert_eq!(batch_challenges.len(), OUT);

                    let inputs = sources
                        .extension_field_inputs
                        .as_chunks::<IN>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();
                    let challenges = batch_challenges
                        .as_chunks::<OUT>()
                        .0
                        .iter()
                        .next()
                        .unwrap_unchecked();

                    for index in 0..accumulator.len() {
                        let value = kernel.evaluate::<_, false>(index, inputs, challenges);
                        for i in 0..2 {
                            accumulator[index][i].add_assign(&value[i]);
                        }
                    }
                }
                for source in sources.extension_field_inputs.iter() {
                    dbg!(source.previous_values());
                    dbg!(source.current_values());
                }
            }
        }
    }
}
