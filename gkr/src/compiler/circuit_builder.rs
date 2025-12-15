use crate::circuit::{Circuit, CircuitError};
use crate::compiler::node::{Node, Term};
use crate::compiler::scheduler::Scheduler;
use field::Field;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug)]
pub struct CircuitBuilder<F: Field> {
    pub nodes: Vec<Node>,

    /// Table of constants (index 0 = F::ZERO, index 1 = F::ONE)
    pub constants: Vec<F>,

    /// Hash map for constant deduplication: hash(F) -> index
    constant_map: HashMap<u64, usize>,

    /// CSE table for common subexpression elimination: hash(Node) -> node_id
    cse_table: HashMap<u64, usize>,

    pub num_inputs: usize,

    /// Indices of output nodes
    pub output_nodes: Vec<usize>,

    pub cse_eliminations: usize,

    /// Number of wires eliminated by DCE (dead code elimination)
    pub dce_eliminations: usize,
}

impl<F: Field + Hash> CircuitBuilder<F> {
    pub fn new() -> Self {
        let mut builder = Self {
            nodes: Vec::new(),
            constants: Vec::new(),
            constant_map: HashMap::new(),
            cse_table: HashMap::new(),
            num_inputs: 0,
            output_nodes: Vec::new(),
            cse_eliminations: 0,
            dce_eliminations: 0,
        };

        // Pre-populate constant table with 0 and 1
        let zero_idx = builder.store_constant(F::ZERO);
        let one_idx = builder.store_constant(F::ONE);

        debug_assert_eq!(zero_idx, 0);
        debug_assert_eq!(one_idx, 1);

        // node 0: wire carrying constant 1 (for linear terms)
        let wire_0 = builder.push_node_internal(Node::input_with_desired_wire_id(0));
        debug_assert_eq!(wire_0, 0);
        builder.num_inputs += 1;

        // mark wire_0 as always needed since copy wires depend on it
        builder.nodes[0].info.is_needed = true;

        builder
    }

    /// store a constant and return its index
    pub fn store_constant(&mut self, c: F) -> usize {
        let hash = self.hash_field_element(&c);

        if let Some(&idx) = self.constant_map.get(&hash) {
            if self.constants[idx] == c {
                return idx;
            }
        }

        // not found, add new constant
        let idx = self.constants.len();
        self.constants.push(c);
        self.constant_map.insert(hash, idx);
        idx
    }

    fn hash_field_element(&self, c: &F) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();

        c.hash(&mut hasher);

        hasher.finish()
    }

    pub fn push_node(&mut self, mut node: Node) -> usize {
        // Compute depth BEFORE CSE check (depth affects equality but not hash)
        node.info.depth = self.compute_node_depth(&node);

        let hash = node.hash();

        if let Some(&existing_id) = self.cse_table.get(&hash) {
            if self.nodes[existing_id] == node {
                // don't count linear terms as CSE eliminations
                if !node.is_linear() {
                    self.cse_eliminations += 1;
                }
                return existing_id;
            }
        }

        self.push_node_internal(node)
    }

    /// push node without CSE check
    fn push_node_internal(&mut self, node: Node) -> usize {
        let node_id = self.nodes.len();
        let hash = node.hash();

        self.nodes.push(node);
        self.cse_table.insert(hash, node_id);

        node_id
    }

    fn compute_node_depth(&self, node: &Node) -> u32 {
        if node.info.is_input {
            return 0;
        }

        let mut depth = 0;
        for term in &node.terms {
            let left_depth = self.nodes[term.left].info.depth;
            let right_depth = self.nodes[term.right].info.depth;

            // +1 because multiplication creates a new layer
            depth = depth.max(1 + left_depth.max(right_depth));
        }
        depth
    }

    pub fn input(&mut self) -> usize {
        let node = Node::input_with_desired_wire_id(self.num_inputs);
        let node_id = self.push_node_internal(node);
        self.num_inputs += 1;
        node_id
    }

    pub fn constant(&mut self, c: F) -> usize {
        if c.is_zero() {
            let node = Node::zero();
            return self.push_node(node);
        }

        let coeff_idx = self.store_constant(c);
        let node = Node::constant(coeff_idx);
        self.push_node(node)
    }

    /// Mark a node as output
    pub fn output(&mut self, node_id: usize) {
        self.nodes[node_id].info.is_output = true;
        self.nodes[node_id].info.desired_wire_id_for_output = Some(self.output_nodes.len());
        self.output_nodes.push(node_id);
    }

    fn get_constant(&self, coeff_idx: usize) -> F {
        self.constants[coeff_idx]
    }

    /// Materialize an input node as a regular term
    fn materialize_input(&self, node_id: usize) -> Node {
        let node = &self.nodes[node_id];

        if node.info.is_input {
            Node::from_term(Term::new(1, 0, node_id))
        } else {
            node.clone()
        }
    }

    /// Create a linear term: 1 * wire_0 * op
    pub fn linear(&mut self, op: usize) -> usize {
        self.mul(op, 0)
    }

    pub fn scale(&mut self, k: F, op: usize) -> usize {
        if k.is_zero() {
            return self.constant(F::ZERO);
        }
        if k.is_one() {
            return op;
        }
        if self.nodes[op as usize].is_zero() {
            return op;
        }
        if self.nodes[op as usize].is_constant() {
            let coeff_idx = self.nodes[op as usize].terms[0].coeff_idx;
            let c = self.get_constant(coeff_idx);
            let mut result = k;
            result.mul_assign(&c);
            return self.constant(result);
        }

        // general case: scale all terms
        let mut node = self.materialize_input(op);
        for term in &mut node.terms {
            let coeff = self.get_constant(term.coeff_idx);
            let mut new_coeff = k;
            new_coeff.mul_assign(&coeff);
            term.coeff_idx = self.store_constant(new_coeff);
        }
        self.push_node(node)
    }

    pub fn mul(&mut self, op0: usize, op1: usize) -> usize {
        self.mul_scaled(F::ONE, op0, op1)
    }

    pub fn mul_scaled(&mut self, mut k: F, mut op0: usize, mut op1: usize) -> usize {
        if k.is_zero() {
            return self.constant(F::ZERO);
        }
        if self.nodes[op0].is_zero() {
            return op0;
        }
        if self.nodes[op1].is_zero() {
            return op1;
        }

        // Constant folding: unwrap constants and linear terms
        // Keep unwrapping until we have non-constant, non-linear operands
        loop {
            let n0 = &self.nodes[op0 as usize];
            let n1 = &self.nodes[op1 as usize];

            // if op0 is constant: absorb into k and multiply k * op1
            if n0.is_constant() {
                let coeff_idx = n0.terms[0].coeff_idx;
                let c = self.get_constant(coeff_idx);
                k.mul_assign(&c);
                return self.scale(k, op1);
            }

            // if op0 is linear: absorb coefficient and unwrap
            // (k1 * wire_0 * x) * k * op1 → (k1*k) * x * op1
            if n0.is_linear() {
                let term = n0.terms[0];
                let coeff = self.get_constant(term.coeff_idx);
                k.mul_assign(&coeff);
                op0 = term.right;
                continue;
            }

            // if op1 is constant or linear, swap and continue
            if n1.is_constant() || n1.is_linear() {
                std::mem::swap(&mut op0, &mut op1);
                continue;
            }

            // both are non-constant, non-linear: create multiplication
            break;
        }

        let coeff_idx = self.store_constant(k);
        let node = Node::from_term(Term::new(coeff_idx, op0, op1));
        self.push_node(node)
    }

    pub fn add(&mut self, op0: usize, op1: usize) -> usize {
        let n0 = &self.nodes[op0];
        let n1 = &self.nodes[op1];

        if n0.is_zero() {
            return op1;
        }
        if n1.is_zero() {
            return op0;
        }

        // depth balancing: force shallower operand through multiplication layer
        let mut op0 = op0;
        let mut op1 = op1;
        let depth0 = n0.info.depth;
        let depth1 = n1.info.depth;

        if depth0 < depth1 {
            op0 = self.linear(op0);
        } else if depth1 < depth0 {
            op1 = self.linear(op1);
        }

        self.merge_nodes(op0, op1)
    }

    /// Subtract: op0 - op1
    pub fn sub(&mut self, op0: usize, op1: usize) -> usize {
        let mut neg_one = F::ZERO;
        neg_one.sub_assign(&F::ONE);
        let neg_op1 = self.scale(neg_one, op1);
        self.add(op0, neg_op1)
    }

    fn merge_nodes(&mut self, op0: usize, op1: usize) -> usize {
        let n0 = self.materialize_input(op0);
        let n1 = self.materialize_input(op1);

        let mut terms = Vec::new();

        let mut i0 = 0;
        let mut i1 = 0;

        while i0 < n0.terms.len() && i1 < n1.terms.len() {
            let t0 = &n0.terms[i0];
            let t1 = &n1.terms[i1];

            if t0.has_same_indices(t1) {
                let c0 = self.get_constant(t0.coeff_idx);
                let c1 = self.get_constant(t1.coeff_idx);
                let mut sum = c0;
                sum.add_assign(&c1);

                if sum != F::ZERO {
                    let sum_idx = self.store_constant(sum);
                    terms.push(Term::new(sum_idx, t0.left, t0.right));
                }
                i0 += 1;
                i1 += 1;
            } else if Self::term_less_than(t0, t1) {
                terms.push(*t0);
                i0 += 1;
            } else {
                terms.push(*t1);
                i1 += 1;
            }
        }

        // add remaining terms
        while i0 < n0.terms.len() {
            terms.push(n0.terms[i0]);
            i0 += 1;
        }
        while i1 < n1.terms.len() {
            terms.push(n1.terms[i1]);
            i1 += 1;
        }

        let node = Node::from_terms(terms);
        self.push_node(node)
    }

    /// Compare terms for ordering (used in merge)
    ///
    /// Terms are ordered lexicographically by (right, left)
    fn term_less_than(t0: &Term, t1: &Term) -> bool {
        if t0.right < t1.right {
            return true;
        }
        if t0.right > t1.right {
            return false;
        }
        t0.left < t1.left
    }

    /// Compute y + a*x
    pub fn axpy(&mut self, y: usize, a: F, x: usize) -> usize {
        if a == F::ZERO {
            return y;
        }
        let ax = self.scale(a, x);
        self.add(y, ax)
    }

    /// Compute y + a
    pub fn apy(&mut self, y: usize, a: F) -> usize {
        if a == F::ZERO {
            return y;
        }
        let c = self.constant(a);
        self.add(y, c)
    }

    pub fn compile(self) -> Result<Circuit<F>, CircuitError> {
        let scheduler = Scheduler::from_builder(self);
        scheduler.schedule()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::Mersenne31Field as F;

    #[test]
    fn test_constant_deduplication() {
        let mut builder = CircuitBuilder::<F>::new();

        let idx1 = builder.store_constant(F::from_nonreduced_u32(42));
        let idx2 = builder.store_constant(F::from_nonreduced_u32(42));

        assert_eq!(idx1, idx2, "Same constant should return same index");
        assert_eq!(builder.constants.len(), 3); // 0, 1, 42
    }

    #[test]
    fn test_input_creation() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();

        assert_eq!(builder.num_inputs, 3); // wire_0, x, y
        assert!(builder.nodes[x as usize].info.is_input);
        assert!(builder.nodes[y as usize].info.is_input);
    }

    #[test]
    fn test_output_marking() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        builder.output(x);

        assert!(builder.nodes[x as usize].info.is_output);
        assert_eq!(builder.output_nodes, vec![x]);
    }

    #[test]
    fn test_materialize_input() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let materialized = builder.materialize_input(x);

        // Should be a linear term: 1 * wire_0 * x
        assert!(materialized.is_linear());
        assert_eq!(materialized.terms[0].coeff_idx, 1);
        assert_eq!(materialized.terms[0].left, 0);
        assert_eq!(materialized.terms[0].right, x);
    }

    #[test]
    fn test_scale() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();

        // Scale by 5
        let scaled = builder.scale(F::from_nonreduced_u32(5), x);
        assert!(!builder.nodes[scaled as usize].is_zero());

        // Scale by 0 should return zero node
        let zero = builder.scale(F::ZERO, x);
        assert!(builder.nodes[zero as usize].is_zero());

        // Scale by 1 should return original
        let same = builder.scale(F::ONE, x);
        assert_eq!(same, x);
    }

    #[test]
    fn test_mul() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();

        let product = builder.mul(x, y);
        let node = &builder.nodes[product as usize];

        // Should have one term: x * y
        assert_eq!(node.terms.len(), 1);
        assert_eq!(node.terms[0].left.min(node.terms[0].right), x);
        assert_eq!(node.terms[0].left.max(node.terms[0].right), y);
    }

    #[test]
    fn test_mul_constant_folding() {
        let mut builder = CircuitBuilder::<F>::new();

        let c1 = builder.constant(F::from_nonreduced_u32(3));
        let c2 = builder.constant(F::from_nonreduced_u32(5));

        let product = builder.mul(c1, c2);

        assert!(builder.nodes[product as usize].is_constant());
        let coeff_idx = builder.nodes[product as usize].terms[0].coeff_idx;
        let result = builder.get_constant(coeff_idx);
        assert_eq!(result, F::from_nonreduced_u32(15));
    }

    #[test]
    fn test_add() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();

        let sum = builder.add(x, y);
        let node = &builder.nodes[sum as usize];

        assert_eq!(node.terms.len(), 2);
    }

    #[test]
    fn test_add_zero() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let zero = builder.constant(F::ZERO);

        let result = builder.add(x, zero);

        assert_eq!(result, x);
    }

    #[test]
    fn test_linear() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let linear_x = builder.linear(x);

        // Should create a term: 1 * wire_0 * (materialized x)
        assert!(linear_x != x);
    }

    #[test]
    fn test_cse_elimination() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.input();
        let y = builder.input();

        // Create same expression twice
        let z1 = builder.mul(x, y);
        let z2 = builder.mul(x, y);

        // Should be the same node due to CSE
        assert_eq!(z1, z2);
        assert_eq!(builder.cse_eliminations, 1);
    }
}
