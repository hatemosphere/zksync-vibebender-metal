use crate::cs::circuit::LookupQueryTableTypeExt;
use crate::definitions::{Degree1Constraint, GKRAddress, Variable};
use crate::gkr_compiler::graph::{graph_element_equals_if_eq, GKRGraph, GraphElement, GraphHolder};
use crate::one_row_compiler::LookupInput;
use crate::tables::TableType;

use super::compiled_constraint::GKRCompiledLinearConstraint;
use super::*;

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct LookupInputNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub inputs: arrayvec::ArrayVec<Degree1Constraint<F>, TOTAL_WIDTH>,
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode for LookupInputNode<F, TOTAL_WIDTH> {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        for c in self.inputs.iter() {
            for (_, v) in c.linear_terms.iter() {
                let idx = graph.get_variable_position_assume_placed(*v);
                dst.push(idx);
            }
        }
    }
}

// this node takes two expressions like 1/(witness + gamma) and will produce 1 new GKR virtual poly
// that is product of denominators
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct TrivialLookupInputDenominatorNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub inputs: [LookupInputNode<F, TOTAL_WIDTH>; 2],
    pub lookup_type: LookupType,
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        for input in self.inputs.iter() {
            input.add_dependencies_into(graph, dst);
        }
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement
    for TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>
{
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        match self.lookup_type {
            LookupType::RangeCheck16 => {
                "Trivial lookup input denominator node in range-check 16".to_string()
            }
            LookupType::TimestampRangeCheck => {
                "Trivial lookup input denominator node in timestamp range-check".to_string()
            }
            LookupType::Generic => {
                "Trivial lookup input denominator node in generic lookup".to_string()
            }
        }
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        // add cached relations
        let parts = self.inputs.each_ref().map(|el| {
            let cached = lookup_input_into_cached_expr(el, &*graph);
            graph.add_cached_relation(cached)
        });

        NoFieldGKRRelation::LookupTrivialDenominator(NoFieldLookupTrivialDenominatorRelation {
            parts,
        })
    }
}

// this node takes two trivial lookup input denominators, and produce single 1 new GKR virtual poly
// that is numerator in LogUp accumulation as (input00 + input01) * denom1 + (input10 + input11) * denom0
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct LookupPostTrivialInputNumeratorNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub trivial_denominator_nodes: [TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>; 2],
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupPostTrivialInputNumeratorNode<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        for input in self.trivial_denominator_nodes.iter() {
            input.add_dependencies_into(graph, dst);
        }
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement
    for LookupPostTrivialInputNumeratorNode<F, TOTAL_WIDTH>
{
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Post trivial lookup input numerator node".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        let parts = self.trivial_denominator_nodes.each_ref().map(|el| {
            let cached_numerators = el.inputs.each_ref().map(|el| {
                let cached = lookup_input_into_cached_expr(el, &*graph);
                graph.add_cached_relation(cached)
            });
            let cached_numerators = NoFieldLookupTrivialDenominatorRelation {
                parts: cached_numerators,
            };
            let denom = graph.get_node_address_assume_placed(el);
            (cached_numerators, denom)
        });

        NoFieldGKRRelation::LookupAggregationPostTrivialNumerator(
            NoFieldLookupPostTrivialNumeratorRelation { parts },
        )
    }
}

// this node takes two trivial lookup input denominators, and produce single 1 new GKR virtual poly
// that is denominator in LogUp accumulation as denom1 * denom0
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct LookupPostTrivialInputDenominatorNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub trivial_denominator_nodes: [TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>; 2],
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupPostTrivialInputDenominatorNode<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        for input in self.trivial_denominator_nodes.iter() {
            input.add_dependencies_into(graph, dst);
        }
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement
    for LookupPostTrivialInputDenominatorNode<F, TOTAL_WIDTH>
{
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        "Post trivial lookup input denominator node".to_string()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        let parts = self
            .trivial_denominator_nodes
            .each_ref()
            .map(|el| graph.get_node_address_assume_placed(el));
        NoFieldGKRRelation::TrivialProduct(parts)
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum NumeratorNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    LinearPartFromTrivialDenominator([LookupInputNode<F, TOTAL_WIDTH>; 2]),
    PostTrivial(LookupPostTrivialInputNumeratorNode<F, TOTAL_WIDTH>),
    Multiplicity(Variable),
    NegativeMultiplicity(Variable),
    Aggregation(Box<LookupAggregationNumerator<F, TOTAL_WIDTH>>),
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode for NumeratorNode<F, TOTAL_WIDTH> {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        match self {
            Self::LinearPartFromTrivialDenominator(inner) => {
                for input in inner.iter() {
                    DependentNode::add_dependencies_into(input, graph, dst);
                }
            }
            Self::PostTrivial(inner) => {
                let node_idx = graph.get_node_index(inner).expect("already placed");
                dst.push(node_idx);
            }
            Self::Multiplicity(var) => {
                let idx = graph.get_variable_position_assume_placed(*var);
                dst.push(idx);
            }
            Self::NegativeMultiplicity(var) => {
                let idx = graph.get_variable_position_assume_placed(*var);
                dst.push(idx);
            }
            Self::Aggregation(inner_box) => {
                let node_idx = graph
                    .get_node_index(inner_box.as_ref().as_dyn())
                    .expect("already placed");
                dst.push(node_idx);
            }
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum DenominatorNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    LinearInput(LookupInputNode<F, TOTAL_WIDTH>),
    TrivialDenominator(TrivialLookupInputDenominatorNode<F, TOTAL_WIDTH>),
    PostTrivial(LookupPostTrivialInputDenominatorNode<F, TOTAL_WIDTH>),
    Aggregation(Box<LookupAggregationDenominator<F, TOTAL_WIDTH>>),
    Setup(arrayvec::ArrayVec<GKRAddress, TOTAL_WIDTH>),
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode for DenominatorNode<F, TOTAL_WIDTH> {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        match self {
            Self::LinearInput(inner) => {
                DependentNode::add_dependencies_into(inner, graph, dst);
            }
            Self::TrivialDenominator(inner) => {
                let node_idx = graph.get_node_index(inner).expect("already placed");
                dst.push(node_idx);
                // DependentNode::add_dependencies_into(inner, graph, dst);
            }
            Self::PostTrivial(inner) => {
                let node_idx = graph.get_node_index(inner).expect("already placed");
                dst.push(node_idx);
            }
            Self::Aggregation(inner_box) => {
                let node_idx = graph
                    .get_node_index(inner_box.as_ref().as_dyn())
                    .expect("already placed");
                dst.push(node_idx);
            }
            Self::Setup(..) => {
                // nothing - assume placed
            }
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum LookupAggregationNode<F: PrimeField, const TOTAL_WIDTH: usize> {
    Recursive {
        lhs: (
            NumeratorNode<F, TOTAL_WIDTH>,
            DenominatorNode<F, TOTAL_WIDTH>,
        ),
        rhs: (
            NumeratorNode<F, TOTAL_WIDTH>,
            DenominatorNode<F, TOTAL_WIDTH>,
        ),
    },
    Join {
        lhs: (
            NumeratorNode<F, TOTAL_WIDTH>,
            DenominatorNode<F, TOTAL_WIDTH>,
        ),
        rhs: LookupInputNode<F, TOTAL_WIDTH>,
    },
    // JoinWithMultiplicity {
    //     lhs_numerator: Variable,
    //     lhs_denominator: LookupInputNode<F, TOTAL_WIDTH>,
    //     rhs: LookupInputNode<F, TOTAL_WIDTH>,
    // }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupAggregationNode<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        match self {
            Self::Recursive { lhs, rhs } => {
                DependentNode::add_dependencies_into(&lhs.0, graph, dst);
                DependentNode::add_dependencies_into(&lhs.1, graph, dst);
                DependentNode::add_dependencies_into(&rhs.0, graph, dst);
                DependentNode::add_dependencies_into(&rhs.1, graph, dst);
            }
            Self::Join { lhs, rhs } => {
                DependentNode::add_dependencies_into(&lhs.0, graph, dst);
                DependentNode::add_dependencies_into(&lhs.1, graph, dst);
                DependentNode::add_dependencies_into(rhs, graph, dst);
            } // Self::JoinWithMultiplicity {
              //     lhs_numerator,
              //     lhs_denominator,
              //     rhs,
              // } => {
              //     let node_idx = graph.get_variable_position_assume_placed(*lhs_numerator);
              //     dst.push(node_idx);
              //     DependentNode::add_dependencies_into(lhs_denominator, graph, dst);
              //     DependentNode::add_dependencies_into(rhs, graph, dst);
              // }
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct LookupAggregationNumerator<F: PrimeField, const TOTAL_WIDTH: usize>(
    pub LookupAggregationNode<F, TOTAL_WIDTH>,
    String,
);

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct LookupAggregationDenominator<F: PrimeField, const TOTAL_WIDTH: usize>(
    pub LookupAggregationNode<F, TOTAL_WIDTH>,
    String,
);

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupAggregationNumerator<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        DependentNode::add_dependencies_into(&self.0, graph, dst);
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> DependentNode
    for LookupAggregationDenominator<F, TOTAL_WIDTH>
{
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        DependentNode::add_dependencies_into(&self.0, graph, dst);
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement
    for LookupAggregationNumerator<F, TOTAL_WIDTH>
{
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        format!("Lookup aggregation numerator node for {}", &self.1)
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        match &self.0 {
            LookupAggregationNode::Recursive {
                lhs: (lhs_num, lhs_den),
                rhs: (rhs_num, rhs_den),
            } => match (lhs_num, lhs_den, rhs_num, rhs_den) {
                (lhs_num, lhs_den, rhs_num, rhs_den) => {
                    panic!(
                        "Combination {:?}/{:?} + {:?}/{:?} is not yet supported",
                        lhs_num, lhs_den, rhs_num, rhs_den
                    );
                }
            },
            LookupAggregationNode::Join { lhs, rhs } => {
                unimplemented!()
            }
        }
    }
}

impl<F: PrimeField, const TOTAL_WIDTH: usize> GraphElement
    for LookupAggregationDenominator<F, TOTAL_WIDTH>
{
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn short_name(&self) -> String {
        format!("Lookup aggregation denominator node for {}", &self.1)
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        todo!()
    }
}

pub(crate) fn layout_width_1_lookup_expressions<F: PrimeField>(
    graph: &mut GKRGraph,
    expressions: Vec<LookupInput<F>>,
    num_variables: &mut u64,
    all_variables_to_place: &mut BTreeSet<Variable>,
    variable_names: &mut HashMap<Variable, String>,
    lookup_type: &str,
    lookup: LookupType,
) -> (
    Variable,
    (
        LookupAggregationNumerator<F, 1>,
        LookupAggregationDenominator<F, 1>,
    ),
) {
    layout_lookup_expressions::<F, 1>(
        graph,
        expressions
            .into_iter()
            .map(|el| {
                (
                    vec![el],
                    LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder),
                )
            })
            .collect(),
        num_variables,
        all_variables_to_place,
        variable_names,
        lookup_type,
        None,
        lookup,
    )
}

fn lookup_input_node_from_expr<F: PrimeField, const TOTAL_WIDTH: usize>(
    expr: &(Vec<LookupInput<F>>, LookupQueryTableTypeExt<F>),
) -> LookupInputNode<F, TOTAL_WIDTH> {
    let (expr, table_type) = expr;
    if TOTAL_WIDTH == 1 {
        assert_eq!(expr.len(), 1);
        assert_eq!(
            *table_type,
            LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
        );
    } else {
        assert!(expr.len() + 1 <= TOTAL_WIDTH)
    }

    let mut inputs = arrayvec::ArrayVec::new();

    for el in expr.iter() {
        match el {
            LookupInput::Variable(var) => {
                inputs.push(Degree1Constraint {
                    linear_terms: vec![(F::ONE, *var)].into_boxed_slice(),
                    constant_term: F::ZERO,
                });
            }
            LookupInput::Expression {
                linear_terms,
                constant_coeff,
            } => {
                inputs.push(Degree1Constraint {
                    linear_terms: linear_terms.clone().into_boxed_slice(),
                    constant_term: *constant_coeff,
                });
            }
        }
    }
    if TOTAL_WIDTH > 1 {
        assert_ne!(
            *table_type,
            LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
        );
        match table_type {
            LookupQueryTableTypeExt::Constant(constant) => {
                inputs.push(Degree1Constraint {
                    linear_terms: vec![].into_boxed_slice(),
                    constant_term: F::from_u64_unchecked(constant.to_table_id() as u64),
                });
            }
            LookupQueryTableTypeExt::Variable(var) => {
                inputs.push(Degree1Constraint {
                    linear_terms: vec![(F::ONE, *var)].into_boxed_slice(),
                    constant_term: F::ZERO,
                });
            }
            LookupQueryTableTypeExt::Expression(expr) => match expr {
                LookupInput::Variable(var) => {
                    inputs.push(Degree1Constraint {
                        linear_terms: vec![(F::ONE, *var)].into_boxed_slice(),
                        constant_term: F::ZERO,
                    });
                }
                LookupInput::Expression {
                    linear_terms,
                    constant_coeff,
                } => {
                    inputs.push(Degree1Constraint {
                        linear_terms: linear_terms.clone().into_boxed_slice(),
                        constant_term: *constant_coeff,
                    });
                }
            },
        }
    }

    LookupInputNode { inputs }
}

pub(crate) fn layout_lookup_expressions<F: PrimeField, const TOTAL_WIDTH: usize>(
    graph: &mut GKRGraph,
    expressions: Vec<(Vec<LookupInput<F>>, LookupQueryTableTypeExt<F>)>,
    num_variables: &mut u64,
    all_variables_to_place: &mut BTreeSet<Variable>,
    variable_names: &mut HashMap<Variable, String>,
    lookup_type: &str,
    final_masking_predicate: Option<Variable>,
    lookup: LookupType,
) -> (
    Variable,
    (
        LookupAggregationNumerator<F, TOTAL_WIDTH>,
        LookupAggregationDenominator<F, TOTAL_WIDTH>,
    ),
) {
    println!(
        "In total of {} lookups of type {}",
        expressions.len(),
        lookup_type
    );

    // create multiplicity
    let multiplicity_var = Variable(*num_variables);
    variable_names.insert(
        multiplicity_var,
        format!("Multiplicity for {}", lookup_type),
    );
    *num_variables += 1;
    all_variables_to_place.insert(multiplicity_var);
    let _ =
        graph.layout_witness_subtree_multiple_variables([multiplicity_var], all_variables_to_place);

    for (expr, table_type) in expressions.iter() {
        if TOTAL_WIDTH == 1 {
            assert_eq!(expr.len(), 1);
            assert_eq!(
                *table_type,
                LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
            );
        } else {
            assert!(expr.len() + 1 <= TOTAL_WIDTH)
        }
    }

    if expressions.len() == 1 {
        if let Some(final_masking_predicate) = final_masking_predicate {
            let last_trivial_input = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&expressions[0]);
            let node = LookupAggregationNode::Recursive {
                lhs: (
                    NumeratorNode::Multiplicity(final_masking_predicate),
                    DenominatorNode::LinearInput(last_trivial_input),
                ),
                rhs: (
                    NumeratorNode::NegativeMultiplicity(multiplicity_var),
                    DenominatorNode::Setup(
                        arrayvec::ArrayVec::try_from(graph.setup_addresses(lookup))
                            .expect("setup must be large enough"),
                    ),
                ),
            };
            let numerator = LookupAggregationNumerator(node.clone(), lookup_type.to_string());
            let denom = LookupAggregationDenominator(node.clone(), lookup_type.to_string());
            graph.add_node(numerator.clone());
            graph.add_node(denom.clone());

            return (multiplicity_var, (numerator, denom));
        } else {
            todo!();
        }
    }

    // our best strategy is to join inputs pairwise, and then glue remainder and multiplicity somewhere

    let mut trivial_denominators = vec![];

    for [a, b] in expressions.as_chunks::<2>().0.iter() {
        let a = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(a);
        let b = lookup_input_node_from_expr::<F, TOTAL_WIDTH>(b);

        let denom = TrivialLookupInputDenominatorNode {
            inputs: [a, b],
            lookup_type: lookup,
        };
        graph.add_node(denom.clone());
        trivial_denominators.push(denom);
    }

    let mut join_node = None;

    if expressions.as_chunks::<2>().1.len() > 0 {
        // create node with multiplicity
        let last_trivial_input =
            lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&expressions.as_chunks::<2>().1[0]);
        let node = LookupAggregationNode::Join {
            lhs: (
                NumeratorNode::NegativeMultiplicity(multiplicity_var),
                DenominatorNode::Setup(
                    arrayvec::ArrayVec::try_from(graph.setup_addresses(lookup))
                        .expect("setup must be large enough"),
                ),
            ),
            rhs: last_trivial_input,
        };
        let numerator = LookupAggregationNumerator(node.clone(), lookup_type.to_string());
        let denom = LookupAggregationDenominator(node.clone(), lookup_type.to_string());
        graph.add_node(numerator.clone());
        graph.add_node(denom.clone());
        join_node = Some((numerator, denom));
    } else {
        // we will need to join multiplicity at the very top
    }

    if trivial_denominators.len() < 2 {
        assert_eq!(trivial_denominators.len(), 1);
        let last = trivial_denominators.pop().unwrap();
        if let Some(join_node) = join_node {
            todo!();
        } else {
            assert!(final_masking_predicate.is_none());
            let last_trivial_input =
                lookup_input_node_from_expr::<F, TOTAL_WIDTH>(&expressions.as_chunks::<2>().1[0]);
            let node = LookupAggregationNode::Join {
                lhs: (
                    NumeratorNode::NegativeMultiplicity(multiplicity_var),
                    DenominatorNode::Setup(
                        arrayvec::ArrayVec::try_from(graph.setup_addresses(lookup))
                            .expect("setup must be large enough"),
                    ),
                ),
                rhs: last_trivial_input,
            };
            let numerator = LookupAggregationNumerator(node.clone(), lookup_type.to_string());
            let denom = LookupAggregationDenominator(node.clone(), lookup_type.to_string());
            graph.add_node(numerator.clone());
            graph.add_node(denom.clone());

            // and join again
            let node = LookupAggregationNode::Recursive {
                lhs: (
                    NumeratorNode::LinearPartFromTrivialDenominator(last.inputs.clone()),
                    DenominatorNode::TrivialDenominator(last),
                ),
                rhs: (
                    NumeratorNode::Aggregation(Box::new(numerator)),
                    DenominatorNode::Aggregation(Box::new(denom)),
                ),
            };
            let numerator = LookupAggregationNumerator(node.clone(), lookup_type.to_string());
            let denom = LookupAggregationDenominator(node.clone(), lookup_type.to_string());
            graph.add_node(numerator.clone());
            graph.add_node(denom.clone());

            return (multiplicity_var, (numerator, denom));
        }
    }

    assert!(trivial_denominators.len() >= 2);

    let mut current_join_nodes = vec![];
    for [a, b] in trivial_denominators.as_chunks::<2>().0.iter() {
        let node = LookupAggregationNode::Recursive {
            lhs: (
                NumeratorNode::LinearPartFromTrivialDenominator(a.inputs.clone()),
                DenominatorNode::TrivialDenominator(a.clone()),
            ),
            rhs: (
                NumeratorNode::LinearPartFromTrivialDenominator(b.inputs.clone()),
                DenominatorNode::TrivialDenominator(b.clone()),
            ),
        };
        let numerator = LookupAggregationNumerator(node.clone(), lookup_type.to_string());
        let denom = LookupAggregationDenominator(node.clone(), lookup_type.to_string());
        graph.add_node(numerator.clone());
        graph.add_node(denom.clone());

        current_join_nodes.push((numerator, denom));
    }
    let mut current_remaining_join_node: Option<(
        LookupAggregationNumerator<F, TOTAL_WIDTH>,
        LookupAggregationDenominator<F, TOTAL_WIDTH>,
    )> = None;
    let mut remaining_multiplicity_to_join = None;
    if trivial_denominators.as_chunks::<2>().1.len() > 0 {
        // join right away with multiplicity if needed
        todo!();
    } else {
        remaining_multiplicity_to_join = Some(multiplicity_var);
    }

    let mut next_join_nodes = vec![];
    let mut next_remaining_join_node = None;

    if current_join_nodes.len() > 1 {
        loop {
            // join in pairs, and then try to join from the previous round or drag all the way along
            for [a, b] in current_join_nodes.as_chunks::<2>().0.iter() {
                // trivial join
                let node = LookupAggregationNode::Recursive {
                    lhs: (
                        NumeratorNode::Aggregation(Box::new(a.0.clone())),
                        DenominatorNode::Aggregation(Box::new(a.1.clone())),
                    ),
                    rhs: (
                        NumeratorNode::Aggregation(Box::new(b.0.clone())),
                        DenominatorNode::Aggregation(Box::new(b.1.clone())),
                    ),
                };
                let numerator = LookupAggregationNumerator(node.clone(), lookup_type.to_string());
                let denom = LookupAggregationDenominator(node.clone(), lookup_type.to_string());
                graph.add_node(numerator.clone());
                graph.add_node(denom.clone());

                next_join_nodes.push((numerator, denom));
            }

            if current_join_nodes.as_chunks::<2>().1.len() > 0 {
                // join right away with multiplicity if needed
                todo!();
            } else {
                next_remaining_join_node = current_remaining_join_node;
            }

            current_join_nodes = next_join_nodes;
            current_remaining_join_node = next_remaining_join_node;

            next_join_nodes = vec![];
            next_remaining_join_node = None;

            if current_join_nodes.len() == 1 {
                break;
            }
        }
    }

    assert_eq!(current_join_nodes.len(), 1);

    let last_join_node = current_join_nodes.pop().unwrap();
    if let Some(remaining_join_node) = current_remaining_join_node {
        assert!(remaining_multiplicity_to_join.is_none());
        todo!();
    } else {
        if let Some(remaining_multiplicity_to_join) = remaining_multiplicity_to_join {
            assert_eq!(remaining_multiplicity_to_join, multiplicity_var);
            assert!(final_masking_predicate.is_none());
            let (n, d) = last_join_node;
            let node = LookupAggregationNode::Recursive {
                lhs: (
                    NumeratorNode::NegativeMultiplicity(remaining_multiplicity_to_join),
                    DenominatorNode::Setup(
                        arrayvec::ArrayVec::try_from(graph.setup_addresses(lookup))
                            .expect("setup must be large enough"),
                    ),
                ),
                rhs: (
                    NumeratorNode::Aggregation(Box::new(n)),
                    DenominatorNode::Aggregation(Box::new(d)),
                ),
            };
            let numerator = LookupAggregationNumerator(node.clone(), lookup_type.to_string());
            let denom = LookupAggregationDenominator(node.clone(), lookup_type.to_string());
            graph.add_node(numerator.clone());
            graph.add_node(denom.clone());

            (multiplicity_var, (numerator, denom))
        } else {
            (multiplicity_var, last_join_node)
        }
    }
}

pub struct GKRLookupInput<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub multiplicity: Option<Variable>,
    pub inputs: arrayvec::ArrayVec<Degree1Constraint<F>, TOTAL_WIDTH>,
}

pub struct CompiledGKRLookupInput<F: PrimeField, const TOTAL_WIDTH: usize> {
    pub multiplicity: Option<GKRAddress>,
    pub inputs: arrayvec::ArrayVec<GKRCompiledLinearConstraint<F>, TOTAL_WIDTH>,
}
