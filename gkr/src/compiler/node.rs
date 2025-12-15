use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Term {
    pub coeff_idx: usize,
    pub left: usize,
    pub right: usize,
}

impl Term {
    pub fn new(coeff_idx: usize, left: usize, right: usize) -> Self {
        let (left, right) = if left <= right {
            (left, right)
        } else {
            (right, left)
        };

        Self {
            coeff_idx,
            left,
            right,
        }
    }

    pub fn is_constant(&self) -> bool {
        self.left == 0 && self.right == 0
    }

    pub fn is_linear(&self) -> bool {
        self.left == 0 && self.right != 0
    }

    pub fn has_same_indices(&self, other: &Term) -> bool {
        self.left == other.left && self.right == other.right
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeInfo {
    pub depth: u32,
    pub is_input: bool,
    pub is_output: bool,
    pub is_needed: bool,
    pub is_assert0: bool,
    pub max_needed_depth: u32,
    pub desired_wire_id_for_input: Option<usize>,
    pub desired_wire_id_for_output: Option<usize>,
}

impl Default for NodeInfo {
    fn default() -> Self {
        Self {
            depth: 0,
            is_input: false,
            is_output: false,
            is_needed: false,
            is_assert0: false,
            max_needed_depth: 0,
            desired_wire_id_for_input: None,
            desired_wire_id_for_output: None,
        }
    }
}

impl NodeInfo {
    pub fn desired_wire_id(&self, depth: u32, depth_ub: u32) -> Option<usize> {
        if self.is_input && depth == 0 {
            self.desired_wire_id_for_input
        } else if self.is_output && depth + 1 == depth_ub {
            self.desired_wire_id_for_output
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub terms: Vec<Term>,
    pub info: NodeInfo,
}

impl Node {
    pub fn zero() -> Self {
        Self {
            terms: Vec::new(),
            info: NodeInfo::default(),
        }
    }

    pub fn input_with_desired_wire_id(wire_id: usize) -> Self {
        Self {
            terms: Vec::new(),
            info: NodeInfo {
                is_input: true,
                desired_wire_id_for_input: Some(wire_id),
                ..Default::default()
            },
        }
    }

    pub fn input() -> Self {
        Self {
            terms: Vec::new(),
            info: NodeInfo {
                is_input: true,
                ..Default::default()
            },
        }
    }

    pub fn constant(coeff_idx: usize) -> Self {
        Self {
            terms: vec![Term::new(coeff_idx, 0, 0)],
            info: NodeInfo::default(),
        }
    }

    pub fn from_term(term: Term) -> Self {
        if term.coeff_idx == 0 {
            Self::zero()
        } else {
            Self {
                terms: vec![term],
                info: NodeInfo::default(),
            }
        }
    }

    pub fn from_terms(terms: Vec<Term>) -> Self {
        Self {
            terms,
            info: NodeInfo::default(),
        }
    }

    pub fn is_zero(&self) -> bool {
        !self.info.is_input && self.terms.is_empty()
    }

    pub fn is_constant(&self) -> bool {
        self.terms.len() == 1 && self.terms[0].is_constant()
    }

    pub fn is_linear(&self) -> bool {
        self.terms.len() == 1 && self.terms[0].is_linear()
    }

    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        self.info.is_input.hash(&mut hasher);
        self.info.is_output.hash(&mut hasher);
        self.info.desired_wire_id_for_input.hash(&mut hasher);
        self.info.desired_wire_id_for_output.hash(&mut hasher);
        self.terms.len().hash(&mut hasher);

        for term in &self.terms {
            term.hash(&mut hasher);
        }

        hasher.finish()
    }
}
