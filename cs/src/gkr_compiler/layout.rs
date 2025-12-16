use std::ops::Range;

use crate::gkr_compiler::graph::{GKRGraph, GraphHolder};

use super::*;

impl GKRGraph {
    pub(crate) fn layout_layers(
        &mut self,
    ) -> (
        Vec<Vec<NoFieldGKRRelation>>,
        Vec<NoFieldGKRCacheRelation>,
        BTreeMap<usize, Range<usize>>, // which caches to compute before which layer
    ) {
        let mut resolved_indexes = HashSet::new();
        // all witness and memory assumed resolved. Setup is not part of dependencies for now
        for (_, pos) in self
            .base_layer_memory
            .iter()
            .chain(self.base_layer_witness.iter())
        {
            let node_idx = self.search_address(pos).expect("already placed");
            let unique = resolved_indexes.insert(node_idx);
            assert!(unique);
        }

        let mut layer_index = 0usize;
        let mut cache_hierarchy = BTreeMap::new();
        let mut all_cached_entries = vec![];
        let mut layers = vec![];

        loop {
            let mut to_layout = vec![];
            let mut all_resolved = true;
            for (node_idx, deps) in self.dependencies.iter() {
                if resolved_indexes.contains(node_idx) {
                    continue;
                }
                let mut can_place = true;
                for dep in deps.iter() {
                    if resolved_indexes.contains(dep) == false {
                        can_place = false;
                        all_resolved = false;
                        break;
                    }
                }
                if can_place {
                    to_layout.push(*node_idx);
                }
            }

            if all_resolved {
                break;
            }

            assert!(to_layout.len() > 0);

            let mut cached_entries = vec![];
            let mut layer = vec![];
            for (in_layer_idx, node_idx) in to_layout.into_iter().enumerate() {
                let node = self.all_nodes[node_idx].dyn_clone();

                println!(
                    "Will resolve {} at layer {}",
                    node.as_ref().short_name(),
                    layer_index
                );

                let num_cached = self.cached_relations.len();
                let relation = node.as_ref().as_dyn().evaluation_description(self);
                let new_caches = &self.cached_relations[num_cached..];
                cached_entries.extend_from_slice(new_caches);
                for cached in new_caches.iter() {
                    let _pos = self.get_cached_relation(cached).expect("already placed");
                }

                layer.push(relation);

                let unique = resolved_indexes.insert(node_idx);
                assert!(unique);

                let pos = GKRAddress::InnerLayer {
                    layer: layer_index,
                    offset: in_layer_idx,
                };
                self.rev_mapping.insert(node_idx, pos);
                self.mapping.insert(pos, node_idx);
            }

            let start = all_cached_entries.len();
            all_cached_entries.extend_from_slice(&cached_entries);
            let end = all_cached_entries.len();
            cache_hierarchy.insert(layer_index, start..end);
            layers.push(layer);

            layer_index += 1;
        }

        (layers, all_cached_entries, cache_hierarchy)
    }
}
