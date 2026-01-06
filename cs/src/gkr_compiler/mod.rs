// GKR compiler tries top optimally place variables into base/intermediate layers. There is no simple
// weight function to define optimization goal, but we can not avoid placing all memory related variables
// into the base layer.

use crate::cs::circuit::CircuitOutput;
use crate::definitions::GKRAddress;
use crate::definitions::REGISTER_SIZE;
use crate::gkr_compiler::graph::GraphHolder;
use common_constants::*;
use field::PrimeField;
use std::collections::*;

mod compiled_constraint;
mod family_circuit;
mod graph;
// mod graphviz;
mod layout;
mod lookup;
pub(crate) mod lookup_nodes;
pub(crate) mod memory_like_grand_product;
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
    pub linear_terms: Box<[Box<[(u64, GKRAddress)]>]>,
    pub constants: Box<[u64]>,
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

impl NoFieldLinearRelation {
    pub fn from_single_input(input: GKRAddress) -> Self {
        Self {
            linear_terms: vec![(1, input)].into_boxed_slice(),
            constant: 0,
        }
    }
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
    FormalBaseLayerInput(GKRAddress),
    // PureQuadratic {
    //     input: NoFieldPureQuadraticGKRRelation,
    //     output: GKRAddress,
    // },
    // MaxQuadratic {
    //     input: NoFieldMaxQuadraticGKRRelation,
    //     output: GKRAddress,
    // },

    // Enforces a randomized set of constraints in a form of c1 + alpha * c2 + ...
    // Sorted as: each quadratic term is recorded once (they are in base field), and powers of alpha are recorded
    EnforceConstraintsMaxQuadratic {
        input: NoFieldMaxQuadraticGKRRelation,
    },
    // SpecialConstraintCollapse(NoFieldSpecialConstraintCollapseGKRRelation),
    // LookupTrivialDenominator(NoFieldLookupTrivialDenominatorRelation),
    // LookupAggregationPostTrivialNumerator(NoFieldLookupPostTrivialNumeratorRelation),

    // Copy across GKR layers, relation is a(x) = \sum_y eq(x, y) a(y) formally
    Copy {
        input: GKRAddress,
        output: GKRAddress,
    },

    // Memory-like argument related

    // Computes (memory tuple) * (memory tuple)
    InitialGrandProductFromCaches {
        input: [GKRAddress; 2],
        output: GKRAddress,
    },
    // Computes (memory tuple) * (single scalar in extension)
    UnbalancedGrandProductWithCache {
        input: [GKRAddress; 2],
        output: GKRAddress,
    },
    // Computes (single scalar in extension) * (single scalar in extension)
    TrivialProduct {
        input: [GKRAddress; 2],
        output: GKRAddress,
    },
    // Computes input * mask + 1 * (1 - mask)
    MaskIntoIdentityProduct {
        input: GKRAddress,
        mask: GKRAddress,
        output: GKRAddress,
    },

    // Lookup argument related
    // Computes linear relation and places it into variable in base field
    MaterializedSingleLookupInput {
        input: NoFieldLinearRelation,
        output: GKRAddress,
    },
    // Computes linear relation for vector lookup and places it into variable in extension field
    MaterializedVectorLookupInput {
        input: NoFieldVectorLookupRelation,
        output: GKRAddress,
    },
    // // Expects both inputs to come from caches, and o
    // LookupPairFromCaches {
    //     input: [[GKRAddress; 2]; 2],
    //     output: [GKRAddress; 2],
    // },
    // Expects denominators to be cached, and computes a/b - c/d -> (num, den)
    LookupWithCachedDensAndSetup {
        input: [GKRAddress; 2],
        setup: [GKRAddress; 2],
        output: [GKRAddress; 2],
    },

    // LookupLinearNumeratorFromCaches([GKRAddress; 2]),
    // LookupDenominatorFromCaches([GKRAddress; 2]),

    // 1/(a+gamma) + 1/(b + gamma) where a, b are in base field
    LookupPairFromBaseInputs {
        input: [NoFieldLinearRelation; 2],
        output: [GKRAddress; 2],
    },
    // a/b + 1/(c + gamma) where `c`` is in the base field
    LookupUnbalancedPairWithBaseInputs {
        input: [GKRAddress; 2],
        remainder: NoFieldLinearRelation,
        output: [GKRAddress; 2],
    },
    // 1/(a+gamma) + multiplicity/(setup + gamma) where a is in base field
    LookupFromBaseInputsWithSetup {
        input: NoFieldLinearRelation,
        setup: [GKRAddress; 2],
        output: [GKRAddress; 2],
    },

    // LookupNumeratorFromBaseInputs([NoFieldLinearRelation; 2]),
    // LookupDenominatorFromBaseInputs([NoFieldLinearRelation; 2]),

    // 1/(a+gamma) + 1/(b + gamma) where a, b are in in extension already due to vector nature (no caching)
    LookupPairFromVectorInputs {
        input: [NoFieldVectorLookupRelation; 2],
        output: [GKRAddress; 2],
    },

    // LookupNumeratorFromVectorInputs([NoFieldVectorLookupRelation; 2]),
    // LookupDenominatorFromVectorInputs([NoFieldVectorLookupRelation; 2]),

    // a/b + c/d -> (num, den)
    LookupPair {
        input: [[GKRAddress; 2]; 2],
        output: [GKRAddress; 2],
    },
    // LookupNumeratorContinueAggregation([GKRAddress; 2]),
    // LookupDenominatorContinueAggregation([GKRAddress; 2]),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoFieldVectorLookupRelation(Box<[NoFieldLinearRelation]>);

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NoFieldGKRCacheRelation {
    LongLinear,
    VectorizedLookup(NoFieldVectorLookupRelation),
    MemoryTuple(NoFieldSpecialMemoryContributionRelation),
    VectorizedLookupSetup(Box<[GKRAddress]>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GateArtifacts {
    pub output_layer: usize,
    pub caching_relations: Vec<(usize, NoFieldGKRCacheRelation)>,
    pub enforced_relation: NoFieldGKRRelation,
}

pub trait GKRGate {
    type Output: 'static + Sized;

    fn short_name(&self) -> String;

    fn add_at_layer(&self, graph: &mut impl GraphHolder, output_layer: usize) -> Self::Output;
}
