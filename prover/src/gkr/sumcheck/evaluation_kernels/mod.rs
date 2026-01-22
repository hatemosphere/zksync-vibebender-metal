use crate::gkr::sumcheck::access_and_fold::GKRStorage;
use cs::definitions::GKRAddress;
use field::{FieldExtension, PrimeField};
pub mod batch_constraint_eval_example;
pub mod trivial_product_in_extension;

pub trait EvaluationRepresentation<F: PrimeField, E: FieldExtension<F> + PrimeField>:
    'static + Clone + Copy + core::fmt::Debug + Send + Sync
{
    type CollapseContext: 'static + Clone + Copy + core::fmt::Debug + Send + Sync;
    type CollapsedForm: PrimeField;
    fn collapse(self, ctx: &Self::CollapseContext) -> Self::CollapsedForm;

    fn collapse_for_batch_eval(self, ctx: &Self::CollapseContext, challenge: &E) -> E;

    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self);
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self);
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self);
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E> for () {
    type CollapseContext = ();
    type CollapsedForm = F;
    #[inline(always)]
    fn collapse(self, _ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        F::ZERO
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, _ctx: &Self::CollapseContext, _challenge: &E) -> E {
        E::ZERO
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, _other: &Self) {}
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, _other: &Self) {}
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, _other: &Self) {}
}

#[derive(Clone, Copy, Debug)]
pub struct BaseFieldRepresentation<F: PrimeField>(pub(crate) F);

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E>
    for BaseFieldRepresentation<F>
{
    type CollapseContext = ();
    type CollapsedForm = F;
    #[inline(always)]
    fn collapse(self, _ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        self.0
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, _ctx: &Self::CollapseContext, challenge: &E) -> E {
        let mut result = *challenge;
        result.mul_assign_by_base(&self.0);
        result
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.0.add_assign(&other.0);
    }
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.0.sub_assign(&other.0);
    }
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.0.mul_assign(&other.0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BaseFieldFoldedOnceRepresentation<F: PrimeField> {
    pub(crate) c0: F,
    pub(crate) c1: F,
    pub(crate) computed_r2_coeff: F,
}

impl<F: PrimeField> BaseFieldFoldedOnceRepresentation<F> {
    #[inline(always)]
    pub fn new(c0: F, c1: F) -> Self {
        Self {
            c0,
            c1,
            computed_r2_coeff: F::ZERO,
        }
    }
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E>
    for BaseFieldFoldedOnceRepresentation<F>
{
    type CollapseContext = (E, E);
    type CollapsedForm = E;
    #[inline(always)]
    fn collapse(self, ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        let (mut r, r2) = *ctx;
        let mut result = r2;
        result.mul_assign_by_base(&self.computed_r2_coeff);
        r.mul_assign_by_base(&self.c1);
        result.add_assign(&r);
        result.add_assign_base(&self.c0);

        result
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, ctx: &Self::CollapseContext, challenge: &E) -> E {
        let mut result = self.collapse(ctx);
        result.mul_assign(challenge);
        result
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.c0.add_assign(&other.c0);
        self.c1.add_assign(&other.c1);
        if ASSUME_NO_PRODUCTS_BEFORE == false {
            self.computed_r2_coeff.add_assign(&other.computed_r2_coeff);
        }
    }
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.c0.sub_assign(&other.c0);
        self.c1.sub_assign(&other.c1);
        if ASSUME_NO_PRODUCTS_BEFORE == false {
            self.computed_r2_coeff.sub_assign(&other.computed_r2_coeff);
        }
    }
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        if ASSUME_NO_PRODUCTS_BEFORE == false {
            panic!();
        }
        self.computed_r2_coeff = self.c1;
        self.computed_r2_coeff.mul_assign(&other.c0);
        self.c1.mul_assign(&other.c0);
        let mut tt = self.c0;
        tt.mul_assign(&other.c1);
        self.c1.add_assign(&tt);

        self.c0.mul_assign(&other.c0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExtensionFieldRepresentation<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub(crate) value: E,
    pub(crate) _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> ExtensionFieldRepresentation<F, E> {
    #[inline(always)]
    pub fn new(value: E) -> Self {
        Self {
            value,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E>
    for ExtensionFieldRepresentation<F, E>
{
    type CollapseContext = ();
    type CollapsedForm = E;
    #[inline(always)]
    fn collapse(self, _ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        self.value
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, ctx: &Self::CollapseContext, challenge: &E) -> E {
        let mut result = self.collapse(ctx);
        result.mul_assign(challenge);
        result
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.value.add_assign(&other.value);
    }
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.value.sub_assign(&other.value);
    }
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.value.mul_assign(&other.value);
    }
}

pub trait EvaluationFormStorage<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
    R: EvaluationRepresentation<F, E>,
>
{
    fn dummy() -> Self;
    fn get_collapse_context(&self) -> &R::CollapseContext;
    fn get_f0_only(&self, index: usize) -> R {
        self.get_f0_and_f1_minus_f0(index)[0]
    }
    fn get_f1_minus_f0_only(&self, index: usize) -> R {
        self.get_f0_and_f1_minus_f0(index)[1]
    }
    fn get_f0_and_f1_minus_f0(&self, index: usize) -> [R; 2];
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField, R: EvaluationRepresentation<F, E>>
    EvaluationFormStorage<F, E, R> for ()
{
    fn dummy() -> Self {
        ()
    }
    #[inline(always)]
    fn get_collapse_context(&self) -> &R::CollapseContext {
        unreachable!()
    }
    #[inline(always)]
    fn get_f0_and_f1_minus_f0(&self, _index: usize) -> [R; 2] {
        unreachable!()
    }
}

pub struct SumcheckAccumulatorDst<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub(crate) dest: *mut [E; 2],
    pub(crate) _marker: core::marker::PhantomData<F>,
}

unsafe impl<F: PrimeField, E: FieldExtension<F> + PrimeField> Send
    for SumcheckAccumulatorDst<F, E>
{
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> SumcheckAccumulatorDst<F, E> {
    #[inline(always)]
    pub(crate) fn get_dst(&self, index: usize) -> &mut [E; 2] {
        unsafe { self.dest.add(index).as_mut_unchecked() }
    }
}

pub trait SingleInputTypeBatchSumcheckEvaluationKernel<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
>
{
    fn evaluate_first_round<
        R0: EvaluationRepresentation<F, E>,
        S0: EvaluationFormStorage<F, E, R0>,
        ROUT: EvaluationRepresentation<F, E>,
        SOUT: EvaluationFormStorage<F, E, ROUT>,
    >(
        &self,
        index: usize,
        r0_sources: &[S0],
        output_sources: &[SOUT],
        batch_challenge: &E,
    ) -> [E; 2] {
        self.evaluate::<R0, S0>(index, r0_sources, batch_challenge)
    }

    fn evaluate<R0: EvaluationRepresentation<F, E>, S0: EvaluationFormStorage<F, E, R0>>(
        &self,
        index: usize,
        r0_sources: &[S0],
        batch_challenge: &E,
    ) -> [E; 2];
}

pub trait TwoInputTypesBatchSumcheckEvaluationKernel<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
>
{
    fn evaluate_first_round<
        R0: EvaluationRepresentation<F, E>,
        S0: EvaluationFormStorage<F, E, R0>,
        R1: EvaluationRepresentation<F, E>,
        S1: EvaluationFormStorage<F, E, R1>,
        ROUT: EvaluationRepresentation<F, E>,
        SOUT: EvaluationFormStorage<F, E, ROUT>,
    >(
        &self,
        index: usize,
        r0_sources: &[S0],
        r1_sources: &[S1],
        _output_sources: &[SOUT],
        batch_challenge: &E,
    ) -> [E; 2] {
        self.evaluate::<R0, S0, R1, S1>(index, r0_sources, r1_sources, batch_challenge)
    }

    fn evaluate<
        R0: EvaluationRepresentation<F, E>,
        S0: EvaluationFormStorage<F, E, R0>,
        R1: EvaluationRepresentation<F, E>,
        S1: EvaluationFormStorage<F, E, R1>,
    >(
        &self,
        index: usize,
        r0_sources: &[S0],
        r1_sources: &[S1],
        batch_challenge: &E,
    ) -> [E; 2];
}

pub struct GKRInputs {
    inputs_in_base: Vec<GKRAddress>,
    inputs_in_extension: Vec<GKRAddress>,
    outputs_in_base: Vec<GKRAddress>,
    outputs_in_extension: Vec<GKRAddress>,
}

pub trait BatchedGKRKernel<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    fn get_inputs(&self) -> GKRInputs;
    fn evaluate_over_storage(
        &self,
        storage: &mut GKRStorage<F, E>,
        step: usize,
        batch_challenge: &E,
        folding_challenges: &[E],
        accumulator: &mut [[E; 2]],
    );
}

pub fn evaluate_single_input_kernel_with_base_inputs<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
    K: SingleInputTypeBatchSumcheckEvaluationKernel<F, E>,
>(
    kernel: &K,
    inputs: &GKRInputs,
    storage: &mut GKRStorage<F, E>,
    step: usize,
    batch_challenge: &E,
    folding_challenges: &[E],
    accumulator: &mut [[E; 2]],
) {
    // parallelize eventually
    match step {
        0 => {
            let sources = storage.select_for_first_round(inputs);
            assert!(sources.extension_field_inputs.is_empty());
            if sources.base_field_outputs.is_empty() == false {
                for index in 0..accumulator.len() {
                    let value = kernel.evaluate_first_round(
                        index,
                        &sources.base_field_inputs,
                        &sources.base_field_outputs,
                        batch_challenge,
                    );
                    for i in 0..2 {
                        accumulator[index][i].add_assign(&value[i]);
                    }
                }
            } else if sources.extension_field_outputs.is_empty() == false {
                for index in 0..accumulator.len() {
                    let value = kernel.evaluate_first_round(
                        index,
                        &sources.base_field_inputs,
                        &sources.extension_field_outputs,
                        batch_challenge,
                    );
                    for i in 0..2 {
                        accumulator[index][i].add_assign(&value[i]);
                    }
                }
            } else {
                for index in 0..accumulator.len() {
                    let value =
                        kernel.evaluate::<_, _>(index, &sources.base_field_inputs, batch_challenge);
                    for i in 0..2 {
                        accumulator[index][i].add_assign(&value[i]);
                    }
                }
            }
        }
        1 => {
            let sources = storage.select_for_second_round(inputs, folding_challenges);
            assert!(sources.base_field_inputs.is_empty());
            for index in 0..accumulator.len() {
                let value =
                    kernel.evaluate::<_, _>(index, &sources.base_field_inputs, batch_challenge);
                for i in 0..2 {
                    accumulator[index][i].add_assign(&value[i]);
                }
            }
        }
        2 => {
            todo!()
        }
        3.. => {
            todo!()
        }
    }
}

pub fn evaluate_single_input_kernel_with_extension_inputs<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
    K: SingleInputTypeBatchSumcheckEvaluationKernel<F, E>,
>(
    kernel: &K,
    inputs: &GKRInputs,
    storage: &mut GKRStorage<F, E>,
    step: usize,
    batch_challenge: &E,
    folding_challenges: &[E],
    accumulator: &mut [[E; 2]],
) {
    // parallelize eventually
    match step {
        0 => {
            let sources = storage.select_for_first_round(inputs);
            assert!(sources.base_field_inputs.is_empty());
            assert!(sources.base_field_outputs.is_empty());
            if sources.extension_field_outputs.is_empty() == false {
                for index in 0..accumulator.len() {
                    let value = kernel.evaluate_first_round(
                        index,
                        &sources.extension_field_inputs,
                        &sources.extension_field_outputs,
                        batch_challenge,
                    );
                    for i in 0..2 {
                        accumulator[index][i].add_assign(&value[i]);
                    }
                }
            } else {
                for index in 0..accumulator.len() {
                    let value = kernel.evaluate::<_, _>(
                        index,
                        &sources.extension_field_inputs,
                        batch_challenge,
                    );
                    for i in 0..2 {
                        accumulator[index][i].add_assign(&value[i]);
                    }
                }
            }
        }
        1.. => {
            let sources = storage.select_for_second_round(inputs, folding_challenges);
            assert!(sources.base_field_inputs.is_empty());
            for index in 0..accumulator.len() {
                let value = kernel.evaluate::<_, _>(
                    index,
                    &sources.extension_field_inputs,
                    batch_challenge,
                );
                for i in 0..2 {
                    accumulator[index][i].add_assign(&value[i]);
                }
            }
        }
        2 => {
            todo!()
        }
        3.. => {
            todo!()
        }
    }
}
