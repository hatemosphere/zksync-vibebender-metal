// GKR compiler tries top optimally place variables into base/intermediate layers. There is no simple
// weight function to define optimization goal, but we can not avoid placing all memory related variables
// into the base layer.

use crate::cs::circuit::CircuitOutput;
use crate::definitions::GKRAddress;
use crate::definitions::REGISTER_SIZE;
use common_constants::*;
use field::PrimeField;
use std::collections::*;

mod compiled_constraint;
mod family_circuit;
mod graph;
mod graphviz;
mod layout;
mod lookup;
mod range_check_exprs;
mod utils;

pub use self::compiled_constraint::*;
pub use self::lookup::*;
pub(crate) use self::utils::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookupType {
    RangeCheck16,
    TimestampRangeCheck,
    Generic,
}

#[derive(Default)]
pub struct GKRCompiler<F: PrimeField> {
    _marker: std::marker::PhantomData<F>,
}

#[derive(Clone, Debug, Hash, serde::Serialize, serde::Deserialize)]
pub struct GKRCircuitArtifact<F: PrimeField> {
    pub trace_len: usize,
    pub table_offsets: Vec<u32>,
    pub total_tables_size: usize,

    _marker: core::marker::PhantomData<F>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PureQuadraticGKRRelation<F: PrimeField> {
    pub terms: Box<[(GKRAddress, Box<(F, GKRAddress)>)]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaxQuadraticGKRRelation<F: PrimeField> {
    pub quadratic_terms: Box<[(GKRAddress, Box<(F, GKRAddress)>)]>,
    pub linear_terms: Box<(F, GKRAddress)>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SpecialConstraintCollapseGKRRelation<F: PrimeField> {
    pub predicate: GKRAddress,
    pub remainder_from_quadratic: GKRAddress,
    pub sparse_linear_remainders: Box<[Option<GKRAddress>]>,
    pub sparse_constant_remainders: Box<[F]>,
    pub num_terms: usize,
}

#[derive(Clone, Debug, Hash, serde::Serialize, serde::Deserialize)]
pub enum GKRRelation<F: PrimeField> {
    PureQuadratic(PureQuadraticGKRRelation<F>),
    MaxQuadratic(MaxQuadraticGKRRelation<F>),
    SpecialConstraintCollapse(SpecialConstraintCollapseGKRRelation<F>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldPureQuadraticGKRRelation {
    pub terms: Box<[(GKRAddress, Box<[(u64, GKRAddress)]>)]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldMaxQuadraticGKRRelation {
    pub quadratic_terms: Box<[(GKRAddress, Box<[(u64, GKRAddress)]>)]>,
    pub linear_terms: Box<[(u64, GKRAddress)]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldSpecialConstraintCollapseGKRRelation {
    pub predicate: GKRAddress,
    pub remainder_from_quadratic: GKRAddress,
    pub sparse_linear_remainders: Box<[Box<[(u64, GKRAddress)]>]>,
    pub sparse_constant_remainders: Box<[u64]>,
    pub num_terms: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompiledAddressSpaceRelation {
    Constant(u32),
    Pos(GKRAddress),
    Neg(GKRAddress),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompliedAddress {
    Constant(u32),
    U16Space(GKRAddress),
    U32Space([GKRAddress; 2]),
    U32SpaceSpecialIndirect {
        low_base: GKRAddress,
        low_dynamic_offset: Option<GKRAddress>,
        low_offset: u64,
        high: GKRAddress,
    },
    U32SpaceGeneric([(Box<[(u64, GKRAddress)]>, u64); 2]),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompiledAddressSpaceRelationStrict {
    Constant(u32),
    Is(usize),
    Not(usize),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompliedAddressStrict {
    Constant(u32),
    U16Space(usize),
    U32Space([usize; 2]),
    U32SpaceSpecialIndirect {
        low_base: usize,
        low_dynamic_offset: Option<usize>,
        low_offset: u64,
        high: usize,
    },
    U32SpaceGeneric([(Box<[(u64, usize)]>, u64); 2]),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldSpecialMemoryContributionRelation {
    pub address_space: CompiledAddressSpaceRelationStrict,
    pub address: CompliedAddressStrict,
    pub timestamp: [usize; NUM_TIMESTAMP_COLUMNS_FOR_RAM],
    pub value: [usize; REGISTER_SIZE],
    pub timestamp_offset: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldLinearRelation {
    pub linear_terms: Box<[(u64, GKRAddress)]>,
    pub constant: u64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldLookupTrivialDenominatorRelation {
    pub parts: [GKRAddress; 2],
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldLookupPostTrivialNumeratorRelation {
    pub parts: [(NoFieldLookupTrivialDenominatorRelation, GKRAddress); 2],
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NoFieldGKRRelation {
    PureQuadratic(NoFieldPureQuadraticGKRRelation),
    MaxQuadratic(NoFieldMaxQuadraticGKRRelation),
    SpecialConstraintCollapse(NoFieldSpecialConstraintCollapseGKRRelation),
    LookupTrivialDenominator(NoFieldLookupTrivialDenominatorRelation),
    LookupAggregationPostTrivialNumerator(NoFieldLookupPostTrivialNumeratorRelation),
    InitialGrandProduct([GKRAddress; 2]),
    UnbalancedGrandProduct([GKRAddress; 2]),
    TrivialProduct([GKRAddress; 2]),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NoFieldGKRCacheRelation {
    LongLinear,
    VectorizedLookup(Box<[NoFieldLinearRelation]>),
    MemoryTuple(NoFieldSpecialMemoryContributionRelation),
}
