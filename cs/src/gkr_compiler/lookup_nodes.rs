use crate::cs::circuit::LookupQueryTableTypeExt;
use crate::definitions::gkr::NoFieldSingleColumnLookupRelation;
use crate::definitions::{Degree1Constraint, GKRAddress, Variable};
use crate::gkr_compiler::graph::{CopyNode, GKRGraph, GraphHolder, NodeIndex};
use crate::one_row_compiler::LookupInput;
use crate::tables::TableType;

use super::compiled_constraint::GKRCompiledLinearConstraint;
use super::*;

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookupNumerator {
    Identity,
    Positive(GKRAddress),
    Negative(GKRAddress),
    LinearNumeratorFromInitialAccumulationCaches([GKRAddress; 2]),
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LookupDenominator {
    CopiedBaseInput(GKRAddress),
    CopiedCopiedBaseInput(GKRAddress),
    UseInput(NoFieldSingleColumnLookupRelation),
    UseInputViaCopy(GKRAddress),
    UseVectorInput(NoFieldVectorLookupRelation),
    CopyMaterializedVectorInput(GKRAddress),
    MaterializeBaseInput(NoFieldSingleColumnLookupRelation),
    MaterializeVectorInput(NoFieldVectorLookupRelation),
    Setup(GKRAddress),
    VectorSetup(Box<[GKRAddress]>),
    Explicit(GKRAddress),
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupInputRelation<F: PrimeField> {
    pub inputs: Vec<Degree1Constraint<F>>,
}

// This is just a logical holder of what is a rational pair itself. It's not a node, but can add itself or pair of
// selves into the graph
#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupRationalPair {
    pub num: LookupNumerator,
    pub num_node: Option<GKRAddress>,
    pub den: LookupDenominator,
    pub den_node: Option<GKRAddress>,
    pub lookup_type: LookupType,
}

impl LookupRationalPair {
    pub fn add_single_into_graph(
        this: Self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self, NoFieldGKRRelation) {
        // we consider very limited set of options here
        match (this.num, this.den) {
            (LookupNumerator::Identity, LookupDenominator::MaterializeVectorInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_none());

                let node = MaterializeVectorInputNode(input);
                let (den_node, rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::Explicit(den_node),
                    den_node: Some(den_node),
                    lookup_type: this.lookup_type,
                };

                (r, rel)
            }
            (LookupNumerator::Identity, LookupDenominator::UseInputViaCopy(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_none());

                match input {
                    GKRAddress::BaseLayerMemory(..)
                    | GKRAddress::BaseLayerWitness(..)
                    | GKRAddress::Setup(..) => {}
                    _ => {
                        unreachable!()
                    }
                }

                let node = CopyNode::FromBase(input);
                let (den_node, rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedBaseInput(den_node),
                    den_node: Some(den_node),
                    lookup_type: this.lookup_type,
                };

                (r, rel)
            }
            (LookupNumerator::Identity, LookupDenominator::CopiedBaseInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_some());

                let GKRAddress::InnerLayer { .. } = input else {
                    unreachable!()
                };

                let node = CopyNode::FromIntermediate(input);
                let (den_node, rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedCopiedBaseInput(den_node),
                    den_node: Some(den_node),
                    lookup_type: this.lookup_type,
                };

                (r, rel)
            }
            (LookupNumerator::Identity, LookupDenominator::MaterializeBaseInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_none());

                let node = MaterializeSingleInputNode(input);
                let (den_node, rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedCopiedBaseInput(den_node),
                    den_node: Some(den_node),
                    lookup_type: this.lookup_type,
                };

                (r, rel)
            }
            (LookupNumerator::Identity, LookupDenominator::CopiedCopiedBaseInput(input)) => {
                assert!(this.num_node.is_none());
                assert!(this.den_node.is_some());

                let GKRAddress::InnerLayer { .. } = input else {
                    unreachable!()
                };

                let node = CopyNode::FromIntermediate(input);
                let (den_node, rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Identity,
                    num_node: None,
                    den: LookupDenominator::CopiedCopiedBaseInput(den_node),
                    den_node: Some(den_node),
                    lookup_type: this.lookup_type,
                };

                (r, rel)
            }
            (num, den) => {
                panic!("{:?}/{:?} is not supported", num, den);
            }
        }
    }

    pub fn accumulate_pair_into_graph(
        pair: (Self, Self),
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self, NoFieldGKRRelation) {
        let (a, b) = pair;
        assert_eq!(a.lookup_type, b.lookup_type);
        let lookup_type = a.lookup_type;

        match (a.num, a.den, b.num, b.den) {
            (
                LookupNumerator::Positive(var),
                LookupDenominator::UseVectorInput(input),
                LookupNumerator::Negative(multiplicity),
                LookupDenominator::VectorSetup(setup),
            ) => {
                let node = LookupMaskedWitnessMinusSetupInputNode {
                    mask: var,
                    input: input.clone(),
                    multiplicity,
                    setup,
                };
                let ([num, den], rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Positive(num),
                    num_node: Some(num),
                    den: LookupDenominator::Explicit(den),
                    den_node: Some(den),
                    lookup_type,
                };

                (r, rel)
            }
            (
                LookupNumerator::Identity,
                LookupDenominator::UseVectorInput(input),
                LookupNumerator::Negative(multiplicity),
                LookupDenominator::VectorSetup(setup),
            ) => {
                todo!();
            }
            (
                LookupNumerator::Identity,
                LookupDenominator::UseInput(input),
                LookupNumerator::Negative(multiplicity),
                LookupDenominator::Setup(setup),
            ) => {
                let node = LookupSingleColumnWitnessMinusSetupInputNode {
                    input: input.clone(),
                    multiplicity,
                    setup,
                };
                let ([num, den], rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Positive(num),
                    num_node: Some(num),
                    den: LookupDenominator::Explicit(den),
                    den_node: Some(den),
                    lookup_type,
                };

                (r, rel)
            }
            (
                LookupNumerator::Identity,
                LookupDenominator::UseInput(a),
                LookupNumerator::Identity,
                LookupDenominator::UseInput(b),
            ) => {
                let node = LookupSingleColumnWitnessPairAggregationNode {
                    lhs: a.clone(),
                    rhs: b.clone(),
                };
                let ([num, den], rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Positive(num),
                    num_node: Some(num),
                    den: LookupDenominator::Explicit(den),
                    den_node: Some(den),
                    lookup_type,
                };

                (r, rel)
            }
            (
                LookupNumerator::Positive(a_num),
                LookupDenominator::Explicit(a_den),
                LookupNumerator::Positive(b_num),
                LookupDenominator::Explicit(b_den),
            ) => {
                let node = LookupExplicitPairAggregationNode {
                    lhs_num: a_num,
                    lhs_den: a_den,
                    rhs_num: b_num,
                    rhs_den: b_den,
                };
                let ([num, den], rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Positive(num),
                    num_node: Some(num),
                    den: LookupDenominator::Explicit(den),
                    den_node: Some(den),
                    lookup_type,
                };

                (r, rel)
            }
            (
                LookupNumerator::Positive(a_num),
                LookupDenominator::Explicit(a_den),
                LookupNumerator::Identity,
                LookupDenominator::CopiedCopiedBaseInput(input),
            ) => {
                assert!(a.num_node.is_some());
                assert!(a.den_node.is_some());
                assert!(b.num_node.is_none());
                assert!(b.den_node.is_some());

                let node = LookupExplicitPairWithSingleColumnMaterializedInputAggregationNode {
                    lhs_num: a_num,
                    lhs_den: a_den,
                    base_input: input,
                };
                let ([num, den], rel) = node.add_at_layer(graph, output_layer);

                let r = Self {
                    num: LookupNumerator::Positive(num),
                    num_node: Some(num),
                    den: LookupDenominator::Explicit(den),
                    den_node: Some(den),
                    lookup_type,
                };

                (r, rel)
            }
            (a_num, a_den, b_num, b_den) => {
                panic!(
                    "{:?}/{:?} + {:?}/{:?} is not supported",
                    a_num, a_den, b_num, b_den
                );
            }
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterializeSingleInputNode(pub(crate) NoFieldSingleColumnLookupRelation);

impl GKRGate for MaterializeSingleInputNode {
    type Output = GKRAddress;

    fn short_name(&self) -> String {
        "Materialize single lookup input node".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = graph.add_intermediate_variable_at_layer(output_layer);
        // TODO: decide to cache or not, maybe adaptively based on the number of terms

        let relation = NoFieldGKRRelation::MaterializedSingleLookupInput {
            input: self.0.clone(),
            output,
        };
        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MaterializeVectorInputNode(pub(crate) NoFieldVectorLookupRelation);

impl GKRGate for MaterializeVectorInputNode {
    type Output = GKRAddress;

    fn short_name(&self) -> String {
        "Materialize vector lookup input node".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = graph.add_intermediate_variable_at_layer(output_layer);
        // TODO: decide to cache or not, maybe adaptively based on the number of terms

        let relation = NoFieldGKRRelation::MaterializedVectorLookupInput {
            input: self.0.clone(),
            output,
        };
        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupMaskedWitnessMinusSetupInputNode {
    pub mask: GKRAddress,
    pub input: NoFieldVectorLookupRelation,
    pub multiplicity: GKRAddress,
    pub setup: Box<[GKRAddress]>,
}

impl GKRGate for LookupMaskedWitnessMinusSetupInputNode {
    type Output = [GKRAddress; 2];

    fn short_name(&self) -> String {
        "Mask/vector_input - multiplicity/vector_setup".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = [(); 2].map(|_| graph.add_intermediate_variable_at_layer(output_layer));
        let cached_input = NoFieldGKRCacheRelation::VectorizedLookup(self.input.clone());
        let cached_output = NoFieldGKRCacheRelation::VectorizedLookupSetup(self.setup.clone());
        let cached_input = graph.add_cached_relation(cached_input, output_layer);
        let cached_output = graph.add_cached_relation(cached_output, output_layer);

        let relation = NoFieldGKRRelation::LookupWithCachedDensAndSetup {
            input: [self.mask, cached_input],
            setup: [self.multiplicity, cached_output],
            output,
        };
        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessMinusSetupInputNode {
    pub input: NoFieldSingleColumnLookupRelation,
    pub multiplicity: GKRAddress,
    pub setup: GKRAddress,
}

impl GKRGate for LookupSingleColumnWitnessMinusSetupInputNode {
    type Output = [GKRAddress; 2];

    fn short_name(&self) -> String {
        "1/single_input - multiplicity/single_setup".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = [(); 2].map(|_| graph.add_intermediate_variable_at_layer(output_layer));

        let relation = NoFieldGKRRelation::LookupFromBaseInputsWithSetup {
            input: self.input.clone(),
            setup: [self.multiplicity, self.setup],
            output,
        };

        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupSingleColumnWitnessPairAggregationNode {
    pub lhs: NoFieldSingleColumnLookupRelation,
    pub rhs: NoFieldSingleColumnLookupRelation,
}

impl GKRGate for LookupSingleColumnWitnessPairAggregationNode {
    type Output = [GKRAddress; 2];

    fn short_name(&self) -> String {
        "1/single_input + 1/single_input".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = [(); 2].map(|_| graph.add_intermediate_variable_at_layer(output_layer));

        let relation = NoFieldGKRRelation::LookupPairFromBaseInputs {
            input: [self.lhs.clone(), self.rhs.clone()],
            output,
        };

        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairAggregationNode {
    pub lhs_num: GKRAddress,
    pub lhs_den: GKRAddress,
    pub rhs_num: GKRAddress,
    pub rhs_den: GKRAddress,
}

impl GKRGate for LookupExplicitPairAggregationNode {
    type Output = [GKRAddress; 2];

    fn short_name(&self) -> String {
        "a/b + c/d".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = [(); 2].map(|_| graph.add_intermediate_variable_at_layer(output_layer));

        let relation = NoFieldGKRRelation::LookupPair {
            input: [[self.lhs_num, self.lhs_den], [self.rhs_num, self.rhs_den]],
            output,
        };
        for el in [self.lhs_num, self.lhs_den, self.rhs_num, self.rhs_den] {
            el.assert_as_dependency_for_layer(output_layer);
        }
        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairWithSingleColumnInputAggregationNode {
    pub lhs_num: GKRAddress,
    pub lhs_den: GKRAddress,
    pub base_input: NoFieldSingleColumnLookupRelation,
}

impl GKRGate for LookupExplicitPairWithSingleColumnInputAggregationNode {
    type Output = [GKRAddress; 2];

    fn short_name(&self) -> String {
        "a/b + 1/single_input".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = [(); 2].map(|_| graph.add_intermediate_variable_at_layer(output_layer));

        let relation = NoFieldGKRRelation::LookupUnbalancedPairWithBaseInputs {
            input: [self.lhs_num, self.lhs_den],
            remainder: self.base_input.clone(),
            output,
        };

        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LookupExplicitPairWithSingleColumnMaterializedInputAggregationNode {
    pub lhs_num: GKRAddress,
    pub lhs_den: GKRAddress,
    pub base_input: GKRAddress,
}

impl GKRGate for LookupExplicitPairWithSingleColumnMaterializedInputAggregationNode {
    type Output = [GKRAddress; 2];

    fn short_name(&self) -> String {
        "a/b + 1/single_input".to_string()
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = [(); 2].map(|_| graph.add_intermediate_variable_at_layer(output_layer));

        let relation = NoFieldGKRRelation::LookupUnbalancedPairWithMaterializedBaseInputs {
            input: [self.lhs_num, self.lhs_den],
            remainder: self.base_input,
            output,
        };

        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}
