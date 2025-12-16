use crate::definitions::{GKRAddress, Variable};
use crate::gkr_compiler::{GKRRelation, LookupType, NoFieldGKRCacheRelation, NoFieldGKRRelation};
use std::collections::{BTreeSet, HashSet};
use std::{collections::BTreeMap, hash::Hash};

#[derive(Clone, Copy, Debug)]
struct Placeholder;

impl GraphElement for Placeholder {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        unreachable!()
    }
    fn dependencies(&self, _graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        unreachable!()
    }
    fn equals(&self, _other: &dyn GraphElement) -> bool {
        false
    }
    #[track_caller]
    fn short_name(&self) -> String {
        unreachable!()
    }
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        unreachable!()
    }
}

pub struct GKRGraph {
    pub(crate) all_nodes: Vec<Box<dyn GraphElement>>,
    pub(crate) dependencies: BTreeMap<usize, Vec<usize>>,
    pub(crate) mapping: BTreeMap<GKRAddress, usize>,
    pub(crate) rev_mapping: BTreeMap<usize, GKRAddress>,
    pub(crate) base_layer_memory: BTreeMap<Variable, GKRAddress>,
    pub(crate) base_layer_memory_rev: BTreeMap<GKRAddress, Variable>,
    pub(crate) base_layer_witness: BTreeMap<Variable, GKRAddress>,
    pub(crate) base_layer_witness_rev: BTreeMap<GKRAddress, Variable>,
    pub(crate) setups: Vec<GKRAddress>,
    pub(crate) cached_relations: Vec<NoFieldGKRCacheRelation>,
    pub(crate) range_check_16_setup_column: GKRAddress,
    pub(crate) timestamp_check_16_setup_column: GKRAddress,
    pub(crate) generic_lookup_setup_width: usize,
}

impl GKRGraph {
    pub fn new(generic_lookup_setup_width: usize) -> Self {
        let mut new = Self {
            all_nodes: vec![],
            dependencies: BTreeMap::new(),
            mapping: BTreeMap::new(),
            rev_mapping: BTreeMap::new(),
            base_layer_memory: BTreeMap::new(),
            base_layer_memory_rev: BTreeMap::new(),
            base_layer_witness: BTreeMap::new(),
            base_layer_witness_rev: BTreeMap::new(),
            setups: vec![],
            cached_relations: vec![],
            range_check_16_setup_column: GKRAddress::Setup(0),
            timestamp_check_16_setup_column: GKRAddress::Setup(1),
            generic_lookup_setup_width,
        };

        // add setups as already resolved
        for i in 0..generic_lookup_setup_width {
            new.setups.push(GKRAddress::Setup(2 + i));
        }

        new
    }

    pub(crate) fn search_address(&self, pos: &GKRAddress) -> Option<usize> {
        for (idx, el) in self.all_nodes.iter().enumerate() {
            let other: &dyn GraphElement = el.as_ref();
            if pos.equals(other.as_dyn()) {
                return Some(idx);
            }
        }

        None
    }

    pub(crate) fn search(&self, node: &dyn GraphElement) -> Option<usize> {
        let this: &dyn GraphElement = node.as_dyn();
        for (idx, el) in self.all_nodes.iter().enumerate() {
            let other: &dyn GraphElement = el.as_ref().as_dyn();
            if this.equals(other) {
                return Some(idx);
            }
        }

        None
    }

    fn search_cached_relation(&self, relation: &NoFieldGKRCacheRelation) -> Option<usize> {
        for (idx, el) in self.cached_relations.iter().enumerate() {
            if relation == el {
                return Some(idx);
            }
        }

        None
    }

    fn add_node_impl(&mut self, node: Box<dyn GraphElement>) -> usize {
        let idx = self.all_nodes.len();
        // push placeholder instead
        self.all_nodes.push(Box::new(Placeholder));
        let dependencies = node.dependencies(self);
        let existing = self
            .dependencies
            .insert(idx, dependencies.into_iter().map(|el| el.0).collect());
        assert!(existing.is_none());
        // put actual node
        self.all_nodes[idx] = node;

        idx
    }

    #[track_caller]
    pub(crate) fn layout_memory_subtree_multiple_variables<const N: usize>(
        &mut self,
        variables: [Variable; N],
        all_variables_to_place: &mut BTreeSet<Variable>,
    ) -> [GKRAddress; N] {
        let mut columns = [GKRAddress::placeholder(); N];
        for (i, variable) in variables.into_iter().enumerate() {
            let offset = self.base_layer_memory.len();
            let place = GKRAddress::BaseLayerMemory(offset);
            columns[i] = place;
            self.base_layer_memory.insert(variable, place);
            self.base_layer_memory_rev.insert(place, variable);

            let node_idx = self.all_nodes.len();
            self.all_nodes
                .push(Box::new(place) as Box<dyn GraphElement>);
            // no dependencies
            self.dependencies.insert(node_idx, vec![]);
            self.mapping.insert(place, node_idx);
            self.rev_mapping.insert(node_idx, place);

            assert!(
                all_variables_to_place.remove(&variable),
                "variable {:?} was already placed",
                variable
            );
        }

        columns
    }

    #[track_caller]
    pub(crate) fn layout_witness_subtree_multiple_variables<const N: usize>(
        &mut self,
        variables: [Variable; N],
        all_variables_to_place: &mut BTreeSet<Variable>,
    ) -> [GKRAddress; N] {
        let mut columns = [GKRAddress::placeholder(); N];
        for (i, variable) in variables.into_iter().enumerate() {
            let offset = self.base_layer_witness.len();
            let place = GKRAddress::BaseLayerWitness(offset);
            columns[i] = place;
            self.base_layer_witness.insert(variable, place);
            self.base_layer_witness_rev.insert(place, variable);

            let node_idx = self.all_nodes.len();
            self.all_nodes
                .push(Box::new(place) as Box<dyn GraphElement>);
            // no dependencies
            self.dependencies.insert(node_idx, vec![]);
            self.mapping.insert(place, node_idx);
            self.rev_mapping.insert(node_idx, place);

            assert!(
                all_variables_to_place.remove(&variable),
                "variable {:?} was already placed",
                variable
            );
        }

        columns
    }

    #[track_caller]
    pub(crate) fn get_fixed_layout_pos(&self, variable: &Variable) -> Option<GKRAddress> {
        if let Some(pos) = self.base_layer_memory.get(variable) {
            return Some(*pos);
        }

        if let Some(pos) = self.base_layer_witness.get(variable) {
            return Some(*pos);
        }

        None
    }
}

impl GraphHolder for GKRGraph {
    fn add_node<T: GraphElement>(&mut self, node: T) -> NodeIndex
    where
        Self: Sized,
    {
        // println!("Adding node type {:?}", core::any::type_name::<T>());
        // println!("Adding node {:?}", &node);
        if let Some(index) = self.search(&node) {
            // println!("Node already exists at index {}", index);
            NodeIndex(index)
        } else {
            let added_index = self.add_node_impl(Box::new(node) as Box<dyn GraphElement>);
            // println!("Node added at at index {}", added_index);
            NodeIndex(added_index)
        }
    }

    fn add_node_dyn(&mut self, node: Box<dyn GraphElement>) -> NodeIndex {
        if let Some(index) = self.search(&*node) {
            NodeIndex(index)
        } else {
            let added_index = self.add_node_impl(node);
            NodeIndex(added_index)
        }
    }

    fn get_node_index(&mut self, node: &dyn GraphElement) -> Option<NodeIndex> {
        self.search(node).map(|el| NodeIndex(el))
    }

    fn get_variable_position_assume_placed(&self, variable: Variable) -> NodeIndex {
        let Some(pos) = self.get_fixed_layout_pos(&variable) else {
            panic!("Variable {:?} is not placed", variable);
        };

        let Some(index) = self.search_address(&pos) else {
            panic!(
                "Position {:?} (from variable {:?}) is missing from index",
                pos, variable
            );
        };

        NodeIndex(index)
    }

    fn get_variable_address_assume_placed(&self, variable: Variable) -> GKRAddress {
        let Some(pos) = self.get_fixed_layout_pos(&variable) else {
            panic!("Variable {:?} is not placed", variable);
        };

        pos
    }

    fn get_node_address_assume_placed(&self, node: &dyn GraphElement) -> GKRAddress {
        let node_idx = self.search(node).expect("already placed");
        self.rev_mapping[&node_idx]
    }

    fn add_cached_relation(&mut self, relation: NoFieldGKRCacheRelation) -> GKRAddress {
        if let Some(pos) = self.search_cached_relation(&relation) {
            return GKRAddress::Cached(pos);
        }
        let idx = self.cached_relations.len();
        self.cached_relations.push(relation);

        GKRAddress::Cached(idx)
    }

    fn get_cached_relation(&self, relation: &NoFieldGKRCacheRelation) -> Option<GKRAddress> {
        self.search_cached_relation(&relation)
            .map(|el| GKRAddress::Cached(el))
    }

    fn setup_addresses(&self, lookup_type: LookupType) -> &[GKRAddress] {
        match lookup_type {
            LookupType::RangeCheck16 => std::slice::from_ref(&self.range_check_16_setup_column),
            LookupType::TimestampRangeCheck => {
                std::slice::from_ref(&self.timestamp_check_16_setup_column)
            }
            LookupType::Generic => &self.setups[..],
        }
    }
}

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct NodeIndex(usize);

pub trait GraphHolder {
    fn add_node<T: GraphElement>(&mut self, node: T) -> NodeIndex
    where
        Self: Sized;
    fn add_node_dyn(&mut self, node: Box<dyn GraphElement>) -> NodeIndex;
    fn get_node_index(&mut self, node: &dyn GraphElement) -> Option<NodeIndex>;
    fn add_cached_relation(&mut self, relation: NoFieldGKRCacheRelation) -> GKRAddress;
    fn get_cached_relation(&self, relation: &NoFieldGKRCacheRelation) -> Option<GKRAddress>;
    fn get_variable_position_assume_placed(&self, variable: Variable) -> NodeIndex;
    fn get_variable_address_assume_placed(&self, variable: Variable) -> GKRAddress;
    fn get_node_address_assume_placed(&self, node: &dyn GraphElement) -> GKRAddress;
    fn setup_addresses(&self, lookup_type: LookupType) -> &[GKRAddress];
}

pub trait GraphElement: 'static + core::any::Any + core::fmt::Debug {
    fn downcast_to_self(other: &dyn GraphElement) -> Option<&Self>
    where
        Self: Sized,
    {
        (other as &dyn core::any::Any).downcast_ref::<Self>()
    }
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static);
    fn dyn_clone(&self) -> Box<dyn GraphElement>;
    fn equals(&self, other: &dyn GraphElement) -> bool;
    fn dependencies(&self, graph: &mut dyn GraphHolder) -> Vec<NodeIndex>;
    fn short_name(&self) -> String;
    fn evaluation_description(&self, graph: &mut dyn GraphHolder) -> NoFieldGKRRelation;
}

pub(crate) fn downcast_graph_element<T: Sized + 'static>(other: &dyn GraphElement) -> Option<&T> {
    (other as &dyn core::any::Any).downcast_ref::<T>()
}

pub(crate) fn graph_element_equals_if_eq<T: GraphElement + PartialEq + Eq + 'static>(
    a: &T,
    other: &dyn GraphElement,
) -> bool {
    let other_inner = other.as_dyn();
    if let Some(other) = downcast_graph_element::<T>(other_inner) {
        other == a
    } else {
        false
    }
}

impl GraphElement for GKRAddress {
    fn as_dyn(&'_ self) -> &'_ (dyn GraphElement + 'static) {
        self
    }
    fn dyn_clone(&self) -> Box<dyn GraphElement> {
        Box::new(*self)
    }
    fn equals(&self, other: &dyn GraphElement) -> bool {
        graph_element_equals_if_eq(self, other)
    }
    fn dependencies(&self, _graph: &mut dyn GraphHolder) -> Vec<NodeIndex> {
        // concrete address has no dependencies
        vec![]
    }
    fn short_name(&self) -> String {
        match self {
            Self::BaseLayerMemory(idx) => {
                format!("M{}", idx)
            }
            Self::BaseLayerWitness(idx) => {
                format!("W{}", idx)
            }
            Self::Setup(idx) => {
                format!("S{}", idx)
            }
            Self::InnerLayer { layer, offset } => {
                format!("I{}[{}]", layer, offset)
            }
            Self::OptimizedOut(..) => {
                unreachable!()
            }
            Self::Cached(idx) => {
                format!("C{}", idx)
            }
        }
    }
    #[track_caller]
    fn evaluation_description(&self, _graph: &mut dyn GraphHolder) -> NoFieldGKRRelation {
        unreachable!()
    }
}

#[track_caller]
fn find_variable(
    gkr_address: GKRAddress,
    base_layer_mapping: &BTreeMap<Variable, GKRAddress>,
) -> Option<Variable> {
    match gkr_address {
        GKRAddress::BaseLayerMemory(..) | GKRAddress::BaseLayerWitness(..) => {}
        _ => {
            return None;
        }
    }

    for (var, addr) in base_layer_mapping.iter() {
        if addr == &gkr_address {
            return Some(*var);
        }
    }

    None
}
