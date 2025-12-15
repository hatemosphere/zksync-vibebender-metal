use crate::circuit::{Circuit, CircuitLayer, QuadGate, QuadTerm};
use crate::compiler::circuit_builder::CircuitBuilder;
use crate::compiler::node::Node;
use field::Field;

#[derive(Debug, Clone)]
struct LayerTerm<F: Field> {
    pub coeff: F,
    pub left_wire: u32,
    pub right_wire: u32,
}

#[derive(Debug, Clone)]
struct LayerNode<F: Field> {
    pub terms: Vec<LayerTerm<F>>,
    pub is_copy_wire: bool,
    pub desired_wire_id: Option<u32>,
    pub original_node_id: u32, // For tracking which DAG node this came from
}

impl<F: Field> LayerNode<F> {
    fn new(original_node_id: u32) -> Self {
        Self {
            terms: Vec::new(),
            is_copy_wire: false,
            desired_wire_id: None,
            original_node_id,
        }
    }

    fn copy_wire(from_wire: u32, original_node_id: u32) -> Self {
        Self {
            terms: vec![LayerTerm {
                coeff: F::ONE,
                left_wire: 0,
                right_wire: from_wire,
            }],
            is_copy_wire: true,
            desired_wire_id: None,
            original_node_id,
        }
    }
}

/// Scheduler that converts DAG to layered circuit
pub struct Scheduler<F: Field> {
    nodes: Vec<Node>,
    constants: Vec<F>,
    num_inputs: usize,
    output_nodes: Vec<usize>,
}

impl<F: Field> Scheduler<F> {
    pub fn from_builder(builder: CircuitBuilder<F>) -> Self {
        Self {
            nodes: builder.nodes,
            constants: builder.constants,
            num_inputs: builder.num_inputs,
            output_nodes: builder.output_nodes,
        }
    }

    pub fn schedule(mut self) -> Result<Circuit<F>, crate::circuit::CircuitError> {
        self.compute_needed();

        self.compute_max_needed_depth();

        let depth = self.compute_depth();

        let layers = self.order_by_layer(depth + 1);

        // assign wire IDs and build final circuit
        self.build_circuit(layers)
    }

    /// Mark nodes that are actually needed
    fn compute_needed(&mut self) {
        for output_id in self.output_nodes.clone() {
            self.mark_needed_recursive(output_id);
        }
    }

    fn mark_needed_recursive(&mut self, node_id: usize) {
        let node = &mut self.nodes[node_id as usize];

        if node.info.is_needed {
            return;
        }

        node.info.is_needed = true;

        // Collect all dependencies including node 0 (wire_0)
        let mut deps = Vec::new();
        for term in &node.terms {
            deps.push(term.left);
            deps.push(term.right);
        }

        for dep in deps {
            self.mark_needed_recursive(dep);
        }
    }

    fn compute_max_needed_depth(&mut self) {
        // We need to traverse in topological order
        // For now, simple approach: iterate multiple times until stable
        let mut changed = true;
        while changed {
            changed = false;

            for node_id in 0..self.nodes.len() {
                if !self.nodes[node_id].info.is_needed {
                    continue;
                }

                let mut max_needed = self.nodes[node_id].info.max_needed_depth;

                // Check all nodes that depend on this one
                for other_id in 0..self.nodes.len() {
                    if !self.nodes[other_id].info.is_needed {
                        continue;
                    }

                    let other = &self.nodes[other_id];

                    // Does other depend on node_id?
                    let depends = other
                        .terms
                        .iter()
                        .any(|t| t.left == node_id || t.right == node_id);

                    if depends {
                        max_needed = max_needed.max(other.info.depth);
                    }
                }

                if max_needed > self.nodes[node_id].info.max_needed_depth {
                    self.nodes[node_id].info.max_needed_depth = max_needed;
                    changed = true;
                }
            }
        }

        // Wire_0 (node 0) needs to be copied to all layers where ANY copy wires exist
        // because copy wires compute: wire_0 * value
        if !self.nodes.is_empty() && self.nodes[0].info.is_needed {
            let max_copy_depth = self
                .nodes
                .iter()
                .filter(|n| n.info.is_needed && n.info.max_needed_depth > n.info.depth)
                .map(|n| n.info.max_needed_depth)
                .max()
                .unwrap_or(0);

            self.nodes[0].info.max_needed_depth =
                self.nodes[0].info.max_needed_depth.max(max_copy_depth);
        }
    }

    /// Compute the depth of the circuit (maximum node depth)
    fn compute_depth(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.info.is_needed)
            .map(|n| n.info.depth as usize)
            .max()
            .unwrap_or(0)
    }

    // returns a vector of layers with updated wiring
    fn order_by_layer(&self, depth_ub: usize) -> Vec<Vec<LayerNode<F>>> {
        // 1. Build unsorted layers with lterms using unsorted positions (lops)
        // 2. Sort layers by desired_wire_id
        // 3. Resolution to final wire IDs happens in build_circuit

        let mut layers: Vec<Vec<LayerNode<F>>> = vec![Vec::new(); depth_ub];

        // lops[node_id][depth - node.depth] = unsorted position at that depth
        let mut lops: Vec<Vec<u32>> = vec![Vec::new(); self.nodes.len()];

        for node_id in 0..self.nodes.len() {
            let node = &self.nodes[node_id];

            if !node.info.is_needed || node.is_zero() {
                continue;
            }

            let node_depth = node.info.depth as usize;

            let lop = layers[node_depth].len() as u32;
            lops[node_id].push(lop);

            let mut layer_node = LayerNode::new(node_id as u32);

            for term in &node.terms {
                let coeff = self.constants[term.coeff_idx as usize];

                let dep_depth = if term.left == 0 {
                    0
                } else {
                    self.nodes[term.left as usize].info.depth as usize
                };

                let left_wire = if node_depth == 0 {
                    0
                } else {
                    let offset = (node_depth - 1) - dep_depth;
                    lops[term.left as usize].get(offset).copied().unwrap_or(0)
                };

                let dep_depth_right = if term.right == 0 {
                    0
                } else {
                    self.nodes[term.right as usize].info.depth as usize
                };

                let right_wire = if node_depth == 0 {
                    0
                } else {
                    let offset = (node_depth - 1) - dep_depth_right;
                    lops[term.right as usize].get(offset).copied().unwrap_or(0)
                };

                layer_node.terms.push(LayerTerm {
                    coeff,
                    left_wire,
                    right_wire,
                });
            }

            layer_node.desired_wire_id = node
                .info
                .desired_wire_id(node_depth as u32, depth_ub as u32)
                .map(|x| x as u32);
            layers[node_depth].push(layer_node);

            // create copy wires
            let max_needed = node.info.max_needed_depth as usize;
            for copy_depth in (node_depth + 1)..max_needed {
                let lop_prev = *lops[node_id].last().unwrap();

                let lop_copy = layers[copy_depth].len() as u32;
                lops[node_id].push(lop_copy);

                let mut copy_node = LayerNode::new(node_id as u32);
                copy_node.is_copy_wire = true;
                copy_node.desired_wire_id = node
                    .info
                    .desired_wire_id(copy_depth as u32, depth_ub as u32)
                    .map(|x| x as u32);
                copy_node.terms.push(LayerTerm {
                    coeff: F::ONE,
                    left_wire: 0, // Reference to wire_0's unsorted position (should be 0)
                    right_wire: lop_prev, // Reference to previous copy's unsorted position
                });

                layers[copy_depth].push(copy_node);
            }
        }

        // resolve unsorted positions (lops) to final wire IDs (desired_wire_ids)
        // this must happen BEFORE sorting, while array indices match unsorted positions
        for depth in 1..layers.len() {
            let prev_layer = &layers[depth - 1];

            let wire_mapping: Vec<u32> = prev_layer
                .iter()
                .enumerate()
                .map(|(unsorted_pos, node)| node.desired_wire_id.unwrap_or(unsorted_pos as u32))
                .collect();

            for layer_node in &mut layers[depth] {
                for term in &mut layer_node.terms {
                    term.left_wire = wire_mapping
                        .get(term.left_wire as usize)
                        .copied()
                        .unwrap_or(term.left_wire);
                    term.right_wire = wire_mapping
                        .get(term.right_wire as usize)
                        .copied()
                        .unwrap_or(term.right_wire);
                }
            }
        }

        // sort each layer by desired_wire_id
        // nodes with Some(id) come first sorted by id, then None in insertion order
        for layer in &mut layers {
            layer.sort_by(|a, b| match (a.desired_wire_id, b.desired_wire_id) {
                (Some(x), Some(y)) => x.cmp(&y),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            });
        }

        layers
    }

    fn build_circuit(
        &self,
        layers: Vec<Vec<LayerNode<F>>>,
    ) -> Result<Circuit<F>, crate::circuit::CircuitError> {
        let mut circuit_layers = Vec::new();

        for depth in (1..layers.len()).rev() {
            let layer_nodes = &layers[depth];

            if layer_nodes.is_empty() {
                continue;
            }

            let prev_layer = &layers[depth - 1];

            let mut gates = Vec::new();

            for layer_node in layer_nodes {
                let mut quad_terms = Vec::new();

                for layer_term in &layer_node.terms {
                    quad_terms.push(QuadTerm::new(
                        layer_term.coeff,
                        layer_term.left_wire as usize,
                        layer_term.right_wire as usize,
                    ));
                }

                gates.push(QuadGate::new(quad_terms));
            }

            let circuit_layer = CircuitLayer::new(gates)?;
            circuit_layers.push(circuit_layer);
        }

        Circuit::new(circuit_layers, self.num_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::Mersenne31Field as F;

    #[test]
    fn test_schedule_simple() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();
        let z = builder.mul(x, y);
        builder.output(z);

        let scheduler = Scheduler::from_builder(builder);
        let circuit = scheduler.schedule().unwrap();

        // one multiplication gate
        assert_eq!(circuit.depth(), 1);
    }
}
