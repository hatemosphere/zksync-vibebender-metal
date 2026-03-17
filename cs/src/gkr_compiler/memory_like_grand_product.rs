use super::*;
use crate::cs::circuit_trait::{MemoryAccess, RegisterAccess, WordRepresentation};
use crate::definitions::Variable;

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum GrandProductAccumulationStep {
    Base {
        lhs: MemoryPermutationExpression,
        rhs: MemoryPermutationExpression,
        is_write: bool,
    },
    Aggregation {
        lhs: GKRAddress,
        rhs: GKRAddress,
        is_write: bool,
    },
    Unbalanced {
        lhs: GKRAddress,
        rhs: MemoryPermutationExpression,
        is_write: bool,
    },
}

impl GrandProductAccumulationStep {
    fn is_write(&self) -> bool {
        match self {
            Self::Base { is_write, .. }
            | Self::Aggregation { is_write, .. }
            | Self::Unbalanced { is_write, .. } => *is_write,
        }
    }
}

impl GKRGate for GrandProductAccumulationStep {
    type Output = GKRAddress;

    fn short_name(&self) -> String {
        let is_write = self.is_write();
        match self {
            Self::Base { .. } => {
                if is_write {
                    "Memory grand product base write accumulation node".to_string()
                } else {
                    "Memory grand product base read accumulation node".to_string()
                }
            }
            Self::Aggregation { .. } => {
                if is_write {
                    "Memory grand product aggregation write accumulation node".to_string()
                } else {
                    "Memory grand product aggregation read accumulation node".to_string()
                }
            }
            Self::Unbalanced { .. } => {
                if is_write {
                    "Memory grand product unbalanced write accumulation node".to_string()
                } else {
                    "Memory grand product unbalanced read accumulation node".to_string()
                }
            }
        }
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = graph.add_intermediate_variable_at_layer(output_layer);
        // create caches
        match self {
            Self::Base { lhs, rhs, .. } => {
                let input = [lhs, rhs].map(|el| {
                    let expr = mem_permutation_expr_into_cached_expr(el, graph);
                    assert!(output_layer > 0);
                    let cache_layer = output_layer - 1;
                    graph.add_cached_relation(expr, cache_layer)
                });
                let relation = NoFieldGKRRelation::InitialGrandProductFromCaches { input, output };
                graph.add_enforced_relation(relation.clone(), output_layer);

                (output, relation)
            }
            Self::Aggregation { lhs, rhs, .. } => {
                let input = [*lhs, *rhs];
                let relation = NoFieldGKRRelation::TrivialProduct { input, output };
                graph.add_enforced_relation(relation.clone(), output_layer);

                (output, relation)
            }
            Self::Unbalanced { .. } => {
                todo!();
            }
        }
    }
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GrandProductAccumulationMaskingNode {
    pub lhs: GKRAddress,
    pub mask: GKRAddress,
    pub is_write: bool,
}

impl GKRGate for GrandProductAccumulationMaskingNode {
    type Output = GKRAddress;

    fn short_name(&self) -> String {
        if self.is_write {
            "Grand product write accumulation masking node".to_string()
        } else {
            "Grand product read accumulation masking node".to_string()
        }
    }

    fn add_at_layer(
        &self,
        graph: &mut impl GraphHolder,
        output_layer: usize,
    ) -> (Self::Output, NoFieldGKRRelation) {
        let output = graph.add_intermediate_variable_at_layer(output_layer);
        let relation = NoFieldGKRRelation::MaskIntoIdentityProduct {
            input: self.lhs,
            mask: self.mask,
            output,
        };
        graph.add_enforced_relation(relation.clone(), output_layer);

        (output, relation)
    }
}

pub(crate) fn layout_initial_grand_product_accumulation(
    graph: &mut impl GraphHolder,
    predicate: Variable,
    ram_augmented_sets: &[(MemoryAccess, ShuffleRamTimestampComparisonPartialData)],
    cycle_start_timestamp: [Variable; 2],
    cycle_start_pc: [Variable; 2],
    cycle_end_timestamp: [Variable; 2],
    cycle_end_pc: [Variable; 2],
) -> ((Vec<GKRAddress>, Vec<GKRAddress>), GKRAddress) {
    const PLACEMENT_LAYER: usize = 1;

    let mut grand_product_read_accumulation_nodes = vec![];
    let mut grand_product_write_accumulation_nodes = vec![];

    let copied_predicate_for_grand_product_masking = graph.copy_base_layer_variable(predicate);

    for [a, b] in ram_augmented_sets.as_chunks::<2>().0.iter() {
        // we construct read and write sets separately
        let mut read_set = vec![];
        let mut write_set = vec![];
        for (query, aux) in [a, b] {
            let read_set_el = match query {
                MemoryAccess::RegisterOnly(RegisterAccess { reg_idx, .. }) => {
                    MemoryPermutationExpression {
                        address: AddressSpaceAddress::SingleLimb(*reg_idx),
                        address_space: AddressSpace::Constant(AddressSpaceType::Register),
                        timestamp: aux.read_timestamp,
                        value: query.read_value(),
                        timestamp_offset: 0,
                    }
                }
                MemoryAccess::RegisterOrRam(..) => {
                    todo!();
                } // ShuffleRamQueryType::RegisterOrRam {
                  //     is_register,
                  //     address,
                  // } => {
                  //     let address_space_inner = match is_register {
                  //         Boolean::Is(var) => AddressSpaceIsRegister::Is(var),
                  //         Boolean::Not(var) => AddressSpaceIsRegister::Not(var),
                  //         Boolean::Constant(..) => {
                  //             unreachable!()
                  //         }
                  //     };
                  //     MemoryPermutationExpression {
                  //         address: AddressSpaceAddress::U32Space(address),
                  //         address_space: AddressSpace::RegisterOrRam(address_space_inner),
                  //         timestamp: aux.read_timestamp,
                  //         value: query.read_value,
                  //         timestamp_offset: 0,
                  //     }
                  // }
            };
            read_set.push(read_set_el);

            let write_set_el = match query {
                MemoryAccess::RegisterOnly(RegisterAccess { reg_idx, .. }) => {
                    MemoryPermutationExpression {
                        address: AddressSpaceAddress::SingleLimb(*reg_idx),
                        address_space: AddressSpace::Constant(AddressSpaceType::Register),
                        timestamp: cycle_start_timestamp,
                        value: query.write_value(),
                        timestamp_offset: query.local_timestamp_in_cycle(),
                    }
                }
                MemoryAccess::RegisterOrRam(..) => {
                    todo!();
                } // ShuffleRamQueryType::RegisterOrRam {
                  //     is_register,
                  //     address,
                  // } => {
                  //     let address_space_inner = match is_register {
                  //         Boolean::Is(var) => AddressSpaceIsRegister::Is(var),
                  //         Boolean::Not(var) => AddressSpaceIsRegister::Not(var),
                  //         Boolean::Constant(..) => {
                  //             unreachable!()
                  //         }
                  //     };
                  //     MemoryPermutationExpression {
                  //         address: AddressSpaceAddress::U32Space(address),
                  //         address_space: AddressSpace::RegisterOrRam(address_space_inner),
                  //         timestamp: cycle_start_timestamp,
                  //         value: query.write_value,
                  //         timestamp_offset: query.local_timestamp_in_cycle as u32,
                  //     }
                  // }
            };
            write_set.push(write_set_el);
        }

        let read_set_node = GrandProductAccumulationStep::Base {
            lhs: read_set[0].clone(),
            rhs: read_set[1].clone(),
            is_write: false,
        };
        let (read_set_node, _) = read_set_node.add_at_layer(graph, PLACEMENT_LAYER);
        grand_product_read_accumulation_nodes.push(read_set_node);

        let write_set_node = GrandProductAccumulationStep::Base {
            lhs: write_set[0].clone(),
            rhs: write_set[1].clone(),
            is_write: true,
        };
        let (write_set_node, _) = write_set_node.add_at_layer(graph, PLACEMENT_LAYER);
        grand_product_write_accumulation_nodes.push(write_set_node);
    }

    if ram_augmented_sets.as_chunks::<2>().1.is_empty() == false {
        let last_el = ram_augmented_sets.as_chunks::<2>().1[0].clone();
        // we tread PC permutation as a part of our global permutation under all the same rules

        let mut read_set = vec![];
        let mut write_set = vec![];

        // memory
        {
            let (query, aux) = last_el;
            let read_set_el = match query {
                MemoryAccess::RegisterOnly(RegisterAccess { reg_idx, .. }) => {
                    MemoryPermutationExpression {
                        address: AddressSpaceAddress::SingleLimb(reg_idx),
                        address_space: AddressSpace::Constant(AddressSpaceType::Register),
                        timestamp: aux.read_timestamp,
                        value: query.read_value(),
                        timestamp_offset: 0,
                    }
                }
                MemoryAccess::RegisterOrRam(..) => {
                    todo!();
                }
            };
            read_set.push(read_set_el);

            let write_set_el = match query {
                MemoryAccess::RegisterOnly(RegisterAccess { reg_idx, .. }) => {
                    MemoryPermutationExpression {
                        address: AddressSpaceAddress::SingleLimb(reg_idx),
                        address_space: AddressSpace::Constant(AddressSpaceType::Register),
                        timestamp: cycle_start_timestamp,
                        value: query.write_value(),
                        timestamp_offset: query.local_timestamp_in_cycle(),
                    }
                }
                MemoryAccess::RegisterOrRam(..) => {
                    todo!();
                }
            };
            write_set.push(write_set_el);
        }

        // and PC permutation
        {
            let read_set_el = MemoryPermutationExpression {
                address: AddressSpaceAddress::Empty,
                address_space: AddressSpace::Constant(AddressSpaceType::PC),
                timestamp: cycle_start_timestamp,
                value: WordRepresentation::U16Limbs(cycle_start_pc),
                timestamp_offset: 0,
            };
            read_set.push(read_set_el);

            let write_set_el = MemoryPermutationExpression {
                address: AddressSpaceAddress::Empty,
                address_space: AddressSpace::Constant(AddressSpaceType::PC),
                timestamp: cycle_end_timestamp,
                value: WordRepresentation::U16Limbs(cycle_end_pc),
                timestamp_offset: 0,
            };
            write_set.push(write_set_el);
        }

        let read_set_node = GrandProductAccumulationStep::Base {
            lhs: read_set[0].clone(),
            rhs: read_set[1].clone(),
            is_write: false,
        };
        let (read_set_node, _) = read_set_node.add_at_layer(graph, PLACEMENT_LAYER);
        grand_product_read_accumulation_nodes.push(read_set_node);

        let write_set_node = GrandProductAccumulationStep::Base {
            lhs: write_set[0].clone(),
            rhs: write_set[1].clone(),
            is_write: true,
        };
        let (write_set_node, _) = write_set_node.add_at_layer(graph, PLACEMENT_LAYER);
        grand_product_write_accumulation_nodes.push(write_set_node);
    } else {
        todo!();
    }

    (
        (
            grand_product_read_accumulation_nodes,
            grand_product_write_accumulation_nodes,
        ),
        copied_predicate_for_grand_product_masking,
    )
}

pub(crate) fn accumulate_memory_like_grand_product(
    graph: &mut impl GraphHolder,
    mut copied_predicate_for_grand_product_masking: GKRAddress,
    grand_product_read_accumulation_nodes: Vec<GKRAddress>,
    grand_product_write_accumulation_nodes: Vec<GKRAddress>,
) -> (
    (GKRAddress, NoFieldGKRRelation),
    (GKRAddress, NoFieldGKRRelation),
) {
    let mut next_read_set = vec![];
    let mut next_write_set = vec![];

    copied_predicate_for_grand_product_masking =
        graph.copy_intermediate_layer_variable(copied_predicate_for_grand_product_masking);

    let GKRAddress::InnerLayer { layer, .. } = copied_predicate_for_grand_product_masking else {
        unreachable!()
    };
    let mut placement_layer = layer;
    // accumulation only starts at layer 2
    assert_eq!(placement_layer, 2);

    assert!(grand_product_read_accumulation_nodes.len() > 1);
    assert_eq!(
        grand_product_read_accumulation_nodes.len(),
        grand_product_write_accumulation_nodes.len()
    );

    println!(
        "Continuing grand product accumulation at layer {} for {} contribution pairs",
        placement_layer,
        grand_product_read_accumulation_nodes.len()
    );

    for [a, b] in grand_product_read_accumulation_nodes
        .as_chunks::<2>()
        .0
        .iter()
    {
        let el = GrandProductAccumulationStep::Aggregation {
            lhs: *a,
            rhs: *b,
            is_write: false,
        };
        let el = el.add_at_layer(graph, placement_layer);
        next_read_set.push(el);
    }

    for [a, b] in grand_product_write_accumulation_nodes
        .as_chunks::<2>()
        .0
        .iter()
    {
        let el = GrandProductAccumulationStep::Aggregation {
            lhs: *a,
            rhs: *b,
            is_write: true,
        };
        let el = el.add_at_layer(graph, placement_layer);
        next_write_set.push(el);
    }

    let mut current_read_set = next_read_set;
    let mut current_write_set = next_write_set;
    let mut current_read_remainder = None;
    let mut current_write_remainder = None;

    if grand_product_read_accumulation_nodes
        .as_chunks::<2>()
        .1
        .len()
        > 0
    {
        todo!();
        current_read_remainder =
            Some(grand_product_read_accumulation_nodes.as_chunks::<2>().1[0].clone());
    }
    if grand_product_write_accumulation_nodes
        .as_chunks::<2>()
        .1
        .len()
        > 0
    {
        todo!();
        current_write_remainder =
            Some(grand_product_write_accumulation_nodes.as_chunks::<2>().1[0].clone());
    }

    let mut next_read_set = vec![];
    let mut next_write_set = vec![];
    let mut next_read_remainder = None;
    let mut next_write_remainder = None;

    if current_read_set.len() > 1 || current_write_set.len() > 1 {
        loop {
            copied_predicate_for_grand_product_masking =
                graph.copy_intermediate_layer_variable(copied_predicate_for_grand_product_masking);

            placement_layer += 1;
            let GKRAddress::InnerLayer { layer, .. } = copied_predicate_for_grand_product_masking
            else {
                unreachable!()
            };
            assert_eq!(placement_layer, layer);

            println!(
                "Continuing grand product accumulation at layer {} for {} contribution pairs",
                placement_layer,
                next_read_set.len()
            );

            for [a, b] in current_read_set.as_chunks::<2>().0.iter() {
                let el = GrandProductAccumulationStep::Aggregation {
                    lhs: a.0,
                    rhs: b.0,
                    is_write: false,
                };
                let el = el.add_at_layer(graph, placement_layer);
                next_read_set.push(el);
            }

            for [a, b] in current_write_set.as_chunks::<2>().0.iter() {
                let el = GrandProductAccumulationStep::Aggregation {
                    lhs: a.0,
                    rhs: b.0,
                    is_write: true,
                };
                let el = el.add_at_layer(graph, placement_layer);
                next_write_set.push(el);
            }

            if current_read_set.as_chunks::<2>().1.len() > 0 {
                if let Some(current_read_remainder) = current_read_remainder.take() {
                    let el = GrandProductAccumulationStep::Aggregation {
                        lhs: current_read_set.as_chunks::<2>().1[0].0.clone(),
                        rhs: current_read_remainder,
                        is_write: false,
                    };
                    let el = el.add_at_layer(graph, placement_layer);
                    next_read_set.push(el);
                } else {
                    next_read_remainder = Some(current_read_set.as_chunks::<2>().1[0].0.clone());
                }
            }

            if current_write_set.as_chunks::<2>().1.len() > 0 {
                if let Some(current_write_remainder) = current_write_remainder.take() {
                    let el = GrandProductAccumulationStep::Aggregation {
                        lhs: current_write_set.as_chunks::<2>().1[0].0.clone(),
                        rhs: current_write_remainder,
                        is_write: false,
                    };
                    let el = el.add_at_layer(graph, placement_layer);
                    next_write_set.push(el);
                } else {
                    next_write_remainder = Some(current_write_set.as_chunks::<2>().1[0].0.clone());
                }
            }

            current_read_set = next_read_set;
            current_write_set = next_write_set;
            current_read_remainder = next_read_remainder;
            current_write_remainder = next_write_remainder;

            next_read_set = vec![];
            next_write_set = vec![];
            next_read_remainder = None;
            next_write_remainder = None;
        }
    }

    assert_eq!(current_read_set.len(), 1);
    assert_eq!(current_write_set.len(), 1);
    assert!(current_read_remainder.is_none());
    assert!(current_write_remainder.is_none());

    let read_node = current_read_set.pop().unwrap();
    let write_node = current_write_set.pop().unwrap();

    placement_layer += 1;

    let read_mask = GrandProductAccumulationMaskingNode {
        lhs: read_node.0,
        mask: copied_predicate_for_grand_product_masking,
        is_write: false,
    };
    let read_output = read_mask.add_at_layer(graph, placement_layer);

    let write_mask = GrandProductAccumulationMaskingNode {
        lhs: write_node.0,
        mask: copied_predicate_for_grand_product_masking,
        is_write: true,
    };
    let write_output = write_mask.add_at_layer(graph, placement_layer);

    (read_output, write_output)
}
