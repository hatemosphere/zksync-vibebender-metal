use crate::{
    constraint::Constraint,
    definitions::{GKRAddress, Variable},
    gkr_compiler::graph::{graph_element_equals_if_eq, GKRGraph, GraphElement, GraphHolder},
};

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuadraticConstraintsPartNode<F: PrimeField> {
    pub parts: Vec<Vec<(F, Variable, Variable)>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintsCollapseNode<F: PrimeField> {
    pub predicate: Variable,
    pub quadratic_gate: QuadraticConstraintsPartNode<F>,
    pub linear_parts: Vec<Vec<(F, Variable)>>,
    pub constant_parts: Vec<F>,
}

impl<F: PrimeField> DependentNode for QuadraticConstraintsPartNode<F> {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        // FIXME: handle the case when some variables are indeed intermediate defined via other constraints
        for c in self.parts.iter() {
            for (_, a, b) in c.iter() {
                let a = graph.get_variable_position_assume_placed(*a);
                let b = graph.get_variable_position_assume_placed(*b);
                dst.push(a);
                dst.push(b);
            }
        }
    }
}

impl<F: PrimeField> GraphElement for QuadraticConstraintsPartNode<F> {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn short_name(&self) -> String {
        format!("Quadratic part of {} constraints", self.parts.len())
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        // compute stats
        let mut cross_terms_for_vars = HashMap::<Variable, HashSet<(F, Variable)>>::new();
        for c in self.parts.iter() {
            for (c, a, b) in c.iter() {
                if *a != *b {
                    let p = cross_terms_for_vars.entry(*a).or_default().insert((*c, *b));
                    assert!(p);
                    let p = cross_terms_for_vars.entry(*b).or_default().insert((*c, *a));
                    assert!(p);
                } else {
                    let p = cross_terms_for_vars.entry(*a).or_default().insert((*c, *b));
                    assert!(p);
                }
            }
        }

        // we want to reduce number of multiplications
        let mut selected_terms = vec![];
        loop {
            if cross_terms_for_vars.is_empty() {
                break;
            }
            let (next_best_candidate, _) = cross_terms_for_vars
                .iter()
                .map(|(k, v)| (*k, v.len()))
                .max_by(|a, b| a.1.cmp(&b.1))
                .unwrap();
            let cross_terms = cross_terms_for_vars.remove(&next_best_candidate).unwrap();
            // cleanup
            for (c, other) in cross_terms.iter() {
                if *other != next_best_candidate {
                    let exists = cross_terms_for_vars
                        .get_mut(other)
                        .unwrap()
                        .remove(&(*c, next_best_candidate));
                    assert!(exists);
                    if cross_terms_for_vars.get(other).unwrap().is_empty() {
                        cross_terms_for_vars.remove(other);
                    }
                }
            }
            // we do not care yet about stable sort
            let mut terms = vec![];
            for (c, other) in cross_terms.iter() {
                let place = graph.get_variable_address_assume_placed(*other);
                terms.push((c.as_u64_reduced(), place));
            }
            terms.sort_by(|a, b| a.1.cmp(&b.1));
            let next_best_candidate = graph.get_variable_address_assume_placed(next_best_candidate);
            selected_terms.push((next_best_candidate, terms.into_boxed_slice()));
        }

        NoFieldGKRRelation::PureQuadratic(NoFieldPureQuadraticGKRRelation {
            terms: selected_terms.into_boxed_slice(),
        })
    }
}

impl<F: PrimeField> DependentNode for ConstraintsCollapseNode<F> {
    fn add_dependencies_into(
        &self,
        graph: &mut dyn graph::GraphHolder,
        dst: &mut Vec<graph::NodeIndex>,
    ) {
        dst.push(graph.get_variable_position_assume_placed(self.predicate));
        dst.push(
            graph
                .get_node_index(&self.quadratic_gate)
                .expect("already placed"),
        );

        // FIXME: handle the case when some variables are indeed intermediate defined via other constraints
        for c in self.linear_parts.iter() {
            for (_, a) in c.iter() {
                let a = graph.get_variable_position_assume_placed(*a);
                dst.push(a);
            }
        }
    }
}

impl<F: PrimeField> GraphElement for ConstraintsCollapseNode<F> {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(self.clone())
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn dependencies(&self, graph: &mut dyn graph::GraphHolder) -> Vec<graph::NodeIndex> {
        let mut dst = vec![];
        DependentNode::add_dependencies_into(self, graph, &mut dst);
        dst
    }
    fn short_name(&self) -> String {
        format!(
            "Constraint collapse of {} constraints",
            self.linear_parts.len()
        )
    }

    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        let predicate = graph.get_variable_address_assume_placed(self.predicate);
        let remainder_from_quadratic = graph.get_node_address_assume_placed(&self.quadratic_gate);
        let sparse_constant_remainders = self
            .constant_parts
            .iter()
            .map(|el| el.as_u64_reduced())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let num_terms = sparse_constant_remainders.len();
        let mut sparse_linear_remainders = vec![];
        for set in self.linear_parts.iter() {
            let mut subset = vec![];
            for (c, v) in set.iter() {
                let address = graph.get_variable_address_assume_placed(*v);
                subset.push((c.as_u64_reduced(), address));
            }
            subset.sort_by(|a, b| a.1.cmp(&b.1));
            sparse_linear_remainders.push(subset.into_boxed_slice());
        }

        let sparse_linear_remainders = sparse_linear_remainders.into_boxed_slice();
        assert_eq!(num_terms, sparse_linear_remainders.len());

        NoFieldGKRRelation::SpecialConstraintCollapse(NoFieldSpecialConstraintCollapseGKRRelation {
            predicate,
            remainder_from_quadratic,
            sparse_linear_remainders,
            sparse_constant_remainders,
            num_terms,
        })
    }
}

pub(crate) fn layout_constraints<F: PrimeField>(
    graph: &mut GKRGraph,
    constraints: Vec<(Constraint<F>, bool)>,
    predicate: Variable,
) -> ConstraintsCollapseNode<F> {
    let mut quadratic_parts = vec![];
    let mut linear_parts = vec![];
    let mut constant_parts = vec![];
    for (c, _) in constraints.into_iter() {
        let (q, l, c) = c.split_max_quadratic();
        quadratic_parts.push(q);
        linear_parts.push(l);
        constant_parts.push(c);
    }
    let quadratic_node = QuadraticConstraintsPartNode {
        parts: quadratic_parts,
    };
    graph.add_node(quadratic_node.clone());

    let final_node = ConstraintsCollapseNode {
        predicate,
        linear_parts,
        constant_parts,
        quadratic_gate: quadratic_node,
    };
    graph.add_node(final_node.clone());

    final_node
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GKRCompiledLinearConstraint<F: PrimeField> {
    pub terms: Vec<(F, GKRAddress)>,
    pub constant: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GKRCompiledQuadraticConstraint<F: PrimeField> {
    pub quadratic_terms: Vec<(GKRCompiledLinearConstraint<F>, GKRAddress)>,
    pub linear_terms: Vec<(F, GKRAddress)>,
    pub constant: F,
    pub unique_addresses: BTreeSet<GKRAddress>, // so we know all unique polys to claim about
}
