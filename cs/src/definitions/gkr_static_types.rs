use super::GKRAddress;

/// Output type categories for GKR circuit layers.
#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[repr(u32)]
pub enum OutputType {
    PermutationProduct = 0,
    Lookup16Bits,
    LookupTimestamps,
    GenericLookup,
}

#[derive(Clone, Copy, Debug)]
pub struct StaticNoFieldLinearRelation<'a> {
    pub linear_terms: &'a [(u32, usize)],
    pub constant: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct StaticNoFieldSingleColumnLookupRelation<'a> {
    pub input: StaticNoFieldLinearRelation<'a>,
    pub lookup_set_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct StaticNoFieldVectorLookupRelation<'a> {
    pub columns: &'a [StaticNoFieldLinearRelation<'a>],
    pub lookup_set_index: usize,
}

/// quadratic terms: ((idx_a, idx_b), &[(coeff, power_of_challenge)])
/// linear terms: (idx, &[(coeff, power_of_challenge)])
/// constants: &[(coeff, power_of_challenge)]
#[derive(Clone, Copy, Debug)]
pub struct StaticNoFieldMaxQuadraticConstraintsGKRRelation<'a> {
    pub quadratic_terms: &'a [((usize, usize), &'a [(u32, usize)])],
    pub linear_terms: &'a [(usize, &'a [(u32, usize)])],
    pub constants: &'a [(u32, usize)],
}

#[derive(Clone, Debug)]
#[repr(C, u32)]
pub enum StaticNoFieldGKRRelation<'a> {
    EnforceConstraintsMaxQuadratic {
        input: StaticNoFieldMaxQuadraticConstraintsGKRRelation<'a>,
    },
    Copy {
        input: usize,
        output: usize,
    },
    InitialGrandProductFromCaches {
        input: [usize; 2],
        output: usize,
    },
    UnbalancedGrandProductWithCache {
        scalar: usize,
        input: usize,
        output: usize,
    },
    TrivialProduct {
        input: [usize; 2],
        output: usize,
    },
    MaskIntoIdentityProduct {
        input: usize,
        mask: usize,
        output: usize,
    },
    MaterializeSingleLookupInput {
        input: StaticNoFieldSingleColumnLookupRelation<'a>,
        output: usize,
    },
    MaterializedVectorLookupInput {
        input: StaticNoFieldVectorLookupRelation<'a>,
        output: usize,
    },
    LookupWithCachedDensAndSetup {
        input: [usize; 2],
        setup: [usize; 2],
        output: [usize; 2],
    },
    LookupPairFromBaseInputs {
        input: [StaticNoFieldSingleColumnLookupRelation<'a>; 2],
        output: [usize; 2],
    },
    LookupPairFromMaterializedBaseInputs {
        input: [usize; 2],
        output: [usize; 2],
    },
    LookupUnbalancedPairWithBaseInputs {
        input: [usize; 2],
        remainder: StaticNoFieldSingleColumnLookupRelation<'a>,
        output: [usize; 2],
    },
    LookupFromBaseInputsWithSetup {
        input: StaticNoFieldSingleColumnLookupRelation<'a>,
        setup: [usize; 2],
        output: [usize; 2],
    },
    LookupFromMaterializedBaseInputWithSetup {
        input: usize,
        setup: [usize; 2],
        output: [usize; 2],
    },
    LookupUnbalancedPairWithMaterializedBaseInputs {
        input: [usize; 2],
        remainder: usize,
        output: [usize; 2],
    },
    LookupPairFromVectorInputs {
        input: [StaticNoFieldVectorLookupRelation<'a>; 2],
        output: [usize; 2],
    },
    LookupPair {
        input: [[usize; 2]; 2],
        output: [usize; 2],
    },
}

#[derive(Clone, Debug)]
pub struct StaticGateArtifacts<'a> {
    pub output_layer: usize,
    pub enforced_relation: StaticNoFieldGKRRelation<'a>,
}

#[derive(Clone, Debug)]
pub struct StaticGKRLayerDescription<'a> {
    pub gates: &'a [StaticGateArtifacts<'a>],
    pub gates_with_external_connections: &'a [StaticGateArtifacts<'a>],
    pub additional_base_layer_openings: &'a [GKRAddress],
}
