use super::*;

pub struct ExtensionFieldPoly<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    values: Box<[E]>,
    _marker: core::marker::PhantomData<F>,
}

pub struct ExtensionFieldPolyInitialSource<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    input_start: *mut E,
    next_layer_size: usize,
    _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField>
    EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>
    for ExtensionFieldPolyInitialSource<F, E>
{
    fn dummy() -> Self {
        todo!();
    }
    #[inline(always)]
    fn get_collapse_context(
        &self,
    ) -> &<ExtensionFieldRepresentation<F, E> as EvaluationRepresentation<F, E>>::CollapseContext
    {
        &()
    }
    #[inline(always)]
    fn get_f0_and_f1_minus_f0(&self, index: usize) -> [ExtensionFieldRepresentation<F, E>; 2] {
        // just read and do NOT cache f1 - f0
        todo!();
    }
}

pub struct ExtensionFieldPolyContinuingSource<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    previous_layer_start: *mut E,
    this_layer_start: *mut E,
    this_layer_size: usize,
    next_layer_size: usize,
    _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField>
    EvaluationFormStorage<F, E, ExtensionFieldRepresentation<F, E>>
    for ExtensionFieldPolyContinuingSource<F, E>
{
    fn dummy() -> Self {
        todo!();
    }
    #[inline(always)]
    fn get_collapse_context(
        &self,
    ) -> &<ExtensionFieldRepresentation<F, E> as EvaluationRepresentation<F, E>>::CollapseContext
    {
        &()
    }
    #[inline(always)]
    fn get_f0_and_f1_minus_f0(&self, index: usize) -> [ExtensionFieldRepresentation<F, E>; 2] {
        // read previous, compute and cache, but do not cache f1 - f0 further
        todo!();
    }
}
