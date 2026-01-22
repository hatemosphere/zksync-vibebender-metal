use std::ptr::null_mut;
use std::{collections::BTreeMap, mem::MaybeUninit};

use crate::gkr::sumcheck::evaluation_kernels::BaseFieldFoldedOnceRepresentation;
use crate::gkr::sumcheck::evaluation_kernels::EvaluationFormStorage;
use crate::gkr::sumcheck::evaluation_kernels::EvaluationRepresentation;
use crate::gkr::sumcheck::evaluation_kernels::ExtensionFieldRepresentation;
use crate::gkr::sumcheck::evaluation_kernels::{BaseFieldRepresentation, GKRInputs};
use cs::definitions::GKRAddress;
use field::{FieldExtension, PrimeField};

pub mod input_in_base;
pub mod input_in_extension;

pub use self::input_in_base::*;
pub use self::input_in_extension::*;

pub struct GKRLayerSource<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub layer_idx: usize,
    pub base_field_inputs: BTreeMap<GKRAddress, BaseFieldPoly<F>>,
    pub extension_field_inputs: BTreeMap<GKRAddress, ExtensionFieldPoly<F, E>>,
    pub intermediate_storage_for_folder_base_field_inputs:
        BTreeMap<GKRAddress, BaseFieldPolyIntermediateFoldingStorage<F, E>>,
}

pub struct GKRStorage<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub layers: Vec<GKRLayerSource<F, E>>,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> GKRStorage<F, E> {
    pub fn select_for_first_round(
        &mut self,
        inputs: &GKRInputs,
    ) -> FirstSumcheckRoundSelectedStorage<F, E> {
        todo!()
    }

    pub fn select_for_second_round(
        &mut self,
        inputs: &GKRInputs,
        folding_challenges: &[E],
    ) -> SecondSumcheckRoundSelectedStorage<F, E> {
        todo!()
    }
}

pub struct FirstSumcheckRoundSelectedStorage<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub base_field_inputs: Vec<BaseFieldPolySource<F>>,
    pub extension_field_inputs: Vec<ExtensionFieldPolyInitialSource<F, E>>,
    pub base_field_outputs: Vec<BaseFieldPolySource<F>>,
    pub extension_field_outputs: Vec<ExtensionFieldPolyInitialSource<F, E>>,
    _marker: core::marker::PhantomData<E>,
}

pub struct SecondSumcheckRoundSelectedStorage<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub base_field_inputs: Vec<BaseFieldPolySourceAfterOneFolding<F, E>>,
    pub extension_field_inputs: Vec<ExtensionFieldPolyContinuingSource<F, E>>,
}

pub struct ThirdSumcheckRoundSelectedStorage<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub base_field_inputs: Vec<BaseFieldPolySourceAfterTwoFoldings<F, E>>,
    pub extension_field_inputs: Vec<ExtensionFieldPolyContinuingSource<F, E>>,
}

pub struct FourthAndBeyondSumcheckRoundSelectedStorage<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
> {
    pub base_field_inputs: Vec<()>,
    pub extension_field_inputs: Vec<()>,
    pub base_field_outputs: Vec<()>,
    pub extension_field_outputs: Vec<()>,
    _marker: core::marker::PhantomData<(F, E)>,
}
