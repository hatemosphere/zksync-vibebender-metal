use std::ops::Range;

use crate::gkr_compiler::graph::{GKRGraph, GraphHolder};

use super::*;

pub struct GKRLayout {
    pub layered_relations: Vec<Vec<NoFieldGKRRelation>>,
    pub layered_cached_relations: Vec<Vec<NoFieldGKRCacheRelation>>,
    pub base_layer_vars_participation_in_layers: BTreeMap<GKRAddress, Vec<usize>>, // transitive via cached relations
}

impl GKRGraph {
    pub(crate) fn layout_layers(&mut self) -> (Vec<GateArtifacts>,) {
        // // all witness and memory assumed resolved. Setup is not part of dependencies for now
        // for (_, pos) in self
        //     .base_layer_memory
        //     .iter()
        //     .chain(self.base_layer_witness.iter())
        // {
        //     let node_idx = self.search_address(pos).expect("already placed");
        //     let unique = resolved_indexes.insert(node_idx);
        //     assert!(unique);
        // }

        todo!()
    }
}
