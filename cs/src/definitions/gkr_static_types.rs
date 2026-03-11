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
    pub linear_terms: &'a [(u32, GKRAddress)],
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

/// quadratic terms: ((addr_a, addr_b), &[(coeff, power_of_challenge)])
/// linear terms: (addr, &[(coeff, power_of_challenge)])
/// constants: &[(coeff, power_of_challenge)]
#[derive(Clone, Copy, Debug)]
pub struct StaticNoFieldMaxQuadraticConstraintsGKRRelation<'a> {
    pub quadratic_terms: &'a [((GKRAddress, GKRAddress), &'a [(u32, usize)])],
    pub linear_terms: &'a [(GKRAddress, &'a [(u32, usize)])],
    pub constants: &'a [(u32, usize)],
}

#[derive(Clone, Debug)]
#[repr(C, u32)]
pub enum StaticNoFieldGKRRelation<'a> {
    EnforceConstraintsMaxQuadratic {
        input: StaticNoFieldMaxQuadraticConstraintsGKRRelation<'a>,
    },
    Copy {
        input: GKRAddress,
        output: GKRAddress,
    },
    InitialGrandProductFromCaches {
        input: [GKRAddress; 2],
        output: GKRAddress,
    },
    UnbalancedGrandProductWithCache {
        scalar: GKRAddress,
        input: GKRAddress,
        output: GKRAddress,
    },
    TrivialProduct {
        input: [GKRAddress; 2],
        output: GKRAddress,
    },
    MaskIntoIdentityProduct {
        input: GKRAddress,
        mask: GKRAddress,
        output: GKRAddress,
    },
    MaterializeSingleLookupInput {
        input: StaticNoFieldSingleColumnLookupRelation<'a>,
        output: GKRAddress,
    },
    MaterializedVectorLookupInput {
        input: StaticNoFieldVectorLookupRelation<'a>,
        output: GKRAddress,
    },
    LookupWithCachedDensAndSetup {
        input: [GKRAddress; 2],
        setup: [GKRAddress; 2],
        output: [GKRAddress; 2],
    },
    LookupPairFromBaseInputs {
        input: [StaticNoFieldSingleColumnLookupRelation<'a>; 2],
        output: [GKRAddress; 2],
    },
    LookupPairFromMaterializedBaseInputs {
        input: [GKRAddress; 2],
        output: [GKRAddress; 2],
    },
    LookupUnbalancedPairWithBaseInputs {
        input: [GKRAddress; 2],
        remainder: StaticNoFieldSingleColumnLookupRelation<'a>,
        output: [GKRAddress; 2],
    },
    LookupFromBaseInputsWithSetup {
        input: StaticNoFieldSingleColumnLookupRelation<'a>,
        setup: [GKRAddress; 2],
        output: [GKRAddress; 2],
    },
    LookupFromMaterializedBaseInputWithSetup {
        input: GKRAddress,
        setup: [GKRAddress; 2],
        output: [GKRAddress; 2],
    },
    LookupUnbalancedPairWithMaterializedBaseInputs {
        input: [GKRAddress; 2],
        remainder: GKRAddress,
        output: [GKRAddress; 2],
    },
    LookupPairFromVectorInputs {
        input: [StaticNoFieldVectorLookupRelation<'a>; 2],
        output: [GKRAddress; 2],
    },
    LookupPair {
        input: [[GKRAddress; 2]; 2],
        output: [GKRAddress; 2],
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
