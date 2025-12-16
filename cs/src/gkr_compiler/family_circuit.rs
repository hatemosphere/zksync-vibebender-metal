use super::*;
// use crate::one_row_compiler::*;
// use crate::one_row_compiler::delegation::*;
// use crate::one_row_compiler::compile_layout::*;

use crate::constraint::Constraint;
use crate::constraint::Term;
use crate::cs::circuit::LookupQuery;
use crate::cs::circuit::LookupQueryTableType;
use crate::cs::circuit::LookupQueryTableTypeExt;
use crate::cs::circuit::ShuffleRamQueryType;
use crate::definitions::gkr::*;
use crate::definitions::GKRAddress;
use crate::definitions::Variable;
use crate::gkr_compiler::graph::GKRGraph;
use crate::gkr_compiler::graph::GraphHolder;
use crate::one_row_compiler::compile_layout::ShuffleRamTimestampComparisonPartialData;
use crate::one_row_compiler::delegation::add_compiler_defined_variable;
use crate::one_row_compiler::delegation::add_multiple_compiler_defined_variables;
use crate::one_row_compiler::LookupInput;
use crate::tables::TableType;
use crate::types::Boolean;

impl<F: PrimeField> GKRCompiler<F> {
    pub fn compile_family_circuit(
        &self,
        circuit_output: CircuitOutput<F>,
        max_bytecode_size_in_words: usize,
        num_inits_and_teardowns: usize,
        trace_len_log2: usize,
    ) -> GKRCircuitArtifact<F> {
        assert!(max_bytecode_size_in_words.is_power_of_two());

        let CircuitOutput {
            state_input,
            state_output,
            table_driver,
            num_of_variables,
            constraints,
            lookups,
            shuffle_ram_queries,
            linked_variables,
            range_check_expressions,
            boolean_vars,
            substitutions,
            delegated_computation_requests,
            degegated_request_to_process,
            register_and_indirect_memory_accesses,
            decoder_machine_state,
            executor_machine_state,
            circuit_family_bitmask,
            variable_names,
            variables_from_constraints,
            ..
        } = circuit_output;

        assert!(trace_len_log2 >= TIMESTAMP_COLUMNS_NUM_BITS as usize);

        assert!(state_input.is_empty());
        assert!(state_output.is_empty());
        assert!(linked_variables.is_empty());
        assert!(degegated_request_to_process.is_none());
        assert!(register_and_indirect_memory_accesses.is_empty());
        assert!(decoder_machine_state.is_none());
        assert!(delegated_computation_requests.is_empty());

        let trace_len = 1usize << trace_len_log2;

        let mut variable_names = variable_names;

        // we merge decoder and generic tables as it's always beneficial
        let mut total_tables_size = table_driver.total_tables_len;
        total_tables_size += max_bytecode_size_in_words;

        let lookup_table_encoding_capacity = trace_len;
        let num_required_tuples_for_lookup_setup =
            total_tables_size.div_ceil(lookup_table_encoding_capacity);
        assert!(num_required_tuples_for_lookup_setup <= 1);

        drop(linked_variables);

        let mut constraints = constraints;
        let mut variables_from_constraints = variables_from_constraints;

        let executor_machine_state =
            executor_machine_state.expect("must be present in executor circuit");

        let mut boolean_vars = boolean_vars;
        let mut num_variables = num_of_variables as u64;

        let mut graph = GKRGraph::new(10);

        let mut all_variables_to_place = BTreeSet::new();
        for variable_idx in 0..num_variables {
            all_variables_to_place.insert(Variable(variable_idx));
        }

        let mut range_check_expressions = range_check_expressions;
        // let (shuffle_ram_inits_and_teardowns, lazy_init_aux_set) =
        //     Self::compile_inits_and_teardowns(
        //         num_inits_and_teardowns,
        //         &mut boolean_vars,
        //         &mut range_check_expressions,
        //         &mut num_variables,
        //         &mut memory_tree_offset,
        //         &mut all_variables_to_place,
        //         &mut layout,
        //     );

        let mut shuffle_ram_access_sets = vec![];

        assert!(shuffle_ram_queries.len() < 4);
        assert!(shuffle_ram_queries
            .is_sorted_by(|a, b| a.local_timestamp_in_cycle < b.local_timestamp_in_cycle));
        shuffle_ram_queries.windows(2).for_each(|el| {
            assert!(el[0].local_timestamp_in_cycle + 1 == el[1].local_timestamp_in_cycle)
        });

        let mut shuffle_ram_augmented_sets = vec![];

        for (query_idx, memory_query) in shuffle_ram_queries.clone().into_iter().enumerate() {
            assert_eq!(memory_query.local_timestamp_in_cycle, query_idx);

            let [read_timestamp_low, read_timestamp_high] =
                add_multiple_compiler_defined_variables::<NUM_TIMESTAMP_COLUMNS_FOR_RAM>(
                    &mut num_variables,
                    &mut all_variables_to_place,
                );
            variable_names.insert(
                read_timestamp_low,
                format!("query {}, read_ts[0]", query_idx),
            );
            variable_names.insert(
                read_timestamp_high,
                format!("query {}, read_ts[1]", query_idx),
            );
            let read_timestamp = graph.layout_memory_subtree_multiple_variables(
                [read_timestamp_low, read_timestamp_high],
                &mut all_variables_to_place,
            );

            let borrow_var = {
                // now that we have declared timestamps, we can produce comparison expressions for range checks
                let borrow_var =
                    add_compiler_defined_variable(&mut num_variables, &mut all_variables_to_place);
                boolean_vars.push(borrow_var);
                variable_names.insert(borrow_var, format!("query {}, interm ts borrow", query_idx));

                borrow_var
            };

            let partial_data = ShuffleRamTimestampComparisonPartialData {
                intermediate_borrow: borrow_var,
                read_timestamp: [read_timestamp_low, read_timestamp_high],
                local_timestamp_in_cycle: memory_query.local_timestamp_in_cycle,
            };

            shuffle_ram_augmented_sets.push((memory_query, partial_data));

            let read_value = graph.layout_memory_subtree_multiple_variables(
                memory_query.read_value,
                &mut all_variables_to_place,
            );

            let read_timestamp = read_timestamp.map(|el| {
                let GKRAddress::BaseLayerMemory(el) = el else {
                    unreachable!()
                };

                el
            });

            let read_value = read_value.map(|el| {
                let GKRAddress::BaseLayerMemory(el) = el else {
                    unreachable!()
                };

                el
            });

            let address = match memory_query.query_type {
                ShuffleRamQueryType::RegisterOnly { register_index } => {
                    let [register_index] = graph.layout_memory_subtree_multiple_variables(
                        [register_index],
                        &mut all_variables_to_place,
                    );
                    let GKRAddress::BaseLayerMemory(register_index) = register_index else {
                        unreachable!()
                    };

                    RamAddress::RegisterOnly(RegisterOnlyAccessAddress { register_index })
                }
                ShuffleRamQueryType::RegisterOrRam {
                    is_register,
                    address,
                } => {
                    let is_register = match is_register {
                        Boolean::Is(var) => {
                            let [is_register] = graph.layout_memory_subtree_multiple_variables(
                                [var],
                                &mut all_variables_to_place,
                            );
                            let GKRAddress::BaseLayerMemory(is_register) = is_register else {
                                unreachable!()
                            };
                            IsRegisterAddress::Is(is_register)
                        }
                        Boolean::Not(not_var) => {
                            let [is_not_register] = graph.layout_memory_subtree_multiple_variables(
                                [not_var],
                                &mut all_variables_to_place,
                            );
                            let GKRAddress::BaseLayerMemory(is_not_register) = is_not_register
                            else {
                                unreachable!()
                            };
                            IsRegisterAddress::Not(is_not_register)
                        }
                        Boolean::Constant(..) => {
                            unreachable!()
                        }
                    };
                    let address = graph.layout_memory_subtree_multiple_variables(
                        address,
                        &mut all_variables_to_place,
                    );
                    let address = address.map(|el| {
                        let GKRAddress::BaseLayerMemory(el) = el else {
                            unreachable!()
                        };

                        el
                    });

                    RamAddress::RegisterOrRam(RegisterOrRamAccessAddress {
                        is_register,
                        address,
                    })
                }
            };

            let query_columns = if memory_query.is_readonly() {
                assert_eq!(memory_query.read_value, memory_query.write_value);

                let query_columns = RamReadQuery {
                    in_cycle_write_index: memory_query.local_timestamp_in_cycle as u32,
                    address,
                    read_timestamp,
                    read_value,
                };

                RamQuery::Readonly(query_columns)
            } else {
                let write_value = graph.layout_memory_subtree_multiple_variables(
                    memory_query.write_value,
                    &mut all_variables_to_place,
                );

                let write_value = write_value.map(|el| {
                    let GKRAddress::BaseLayerMemory(el) = el else {
                        unreachable!()
                    };

                    el
                });

                let query_columns = RamWriteQuery {
                    in_cycle_write_index: memory_query.local_timestamp_in_cycle as u32,
                    address,
                    read_timestamp,
                    read_value,
                    write_value,
                };

                RamQuery::Write(query_columns)
            };

            shuffle_ram_access_sets.push(query_columns);
        }

        // we can add explicit nodes for memory access accumulation, and we also explicitly check write timestamps to be in range
        let mut timestamp_range_check_expressions_to_compile = vec![];
        {
            // As all timestamps make a permutation, we only constraint write set
            timestamp_range_check_expressions_to_compile.push(LookupInput::<F>::Variable(
                executor_machine_state.cycle_end_state.timestamp[0],
            ));
            timestamp_range_check_expressions_to_compile.push(LookupInput::<F>::Variable(
                executor_machine_state.cycle_end_state.timestamp[1],
            ));

            // As initial and final timestamps are material and there are range checks on them (output var),
            // we can just do constraints here, as this carry is purely ephemeral

            // We add a constraint (normal one), to perform timestamp += 4 constraint, without
            // carry over top limb, as we want to have upper bound anyway

            // we need to ensure that constraint the describes a carry is boolean
            let mut t = Constraint::from(executor_machine_state.cycle_start_state.timestamp[0])
                + Term::from(TIMESTAMP_STEP)
                - Term::from(executor_machine_state.cycle_end_state.timestamp[0]);
            t.scale(
                F::from_u64_with_reduction(1 << TIMESTAMP_COLUMNS_NUM_BITS)
                    .inverse()
                    .unwrap(),
            );

            // low
            let constraint = t.clone() * t.clone() - t.clone();
            constraints.push((constraint, true));

            // high - carryless
            constraints.push((
                Constraint::from(executor_machine_state.cycle_end_state.timestamp[1])
                    - Term::from(executor_machine_state.cycle_start_state.timestamp[1])
                    - t,
                true,
            ));
        };

        let machine_state = layout_machine_state_for_preprocessed_bytecode(
            &mut graph,
            &mut all_variables_to_place,
            &executor_machine_state,
            circuit_family_bitmask.clone(),
        );

        let mut grand_product_read_accumulation_nodes = vec![];
        let mut grand_product_write_accumulation_nodes = vec![];
        for [a, b] in shuffle_ram_augmented_sets.as_chunks::<2>().0.iter() {
            // we construct read and write sets separately
            let mut read_set = vec![];
            let mut write_set = vec![];
            for (query, aux) in [a, b] {
                let read_set_el = match query.query_type {
                    ShuffleRamQueryType::RegisterOnly { register_index } => {
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::SingleLimb(register_index),
                            address_space: AddressSpace::Constant(AddressSpaceType::Register),
                            timestamp: aux.read_timestamp,
                            value: query.read_value,
                            timestamp_offset: 0,
                        }
                    }
                    ShuffleRamQueryType::RegisterOrRam {
                        is_register,
                        address,
                    } => {
                        let address_space_inner = match is_register {
                            Boolean::Is(var) => AddressSpaceIsRegister::Is(var),
                            Boolean::Not(var) => AddressSpaceIsRegister::Not(var),
                            Boolean::Constant(..) => {
                                unreachable!()
                            }
                        };
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::U32Space(address),
                            address_space: AddressSpace::RegisterOrRam(address_space_inner),
                            timestamp: aux.read_timestamp,
                            value: query.read_value,
                            timestamp_offset: 0,
                        }
                    }
                };
                read_set.push(read_set_el);

                let write_set_el = match query.query_type {
                    ShuffleRamQueryType::RegisterOnly { register_index } => {
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::SingleLimb(register_index),
                            address_space: AddressSpace::Constant(AddressSpaceType::Register),
                            timestamp: executor_machine_state.cycle_start_state.timestamp,
                            value: query.write_value,
                            timestamp_offset: query.local_timestamp_in_cycle as u32,
                        }
                    }
                    ShuffleRamQueryType::RegisterOrRam {
                        is_register,
                        address,
                    } => {
                        let address_space_inner = match is_register {
                            Boolean::Is(var) => AddressSpaceIsRegister::Is(var),
                            Boolean::Not(var) => AddressSpaceIsRegister::Not(var),
                            Boolean::Constant(..) => {
                                unreachable!()
                            }
                        };
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::U32Space(address),
                            address_space: AddressSpace::RegisterOrRam(address_space_inner),
                            timestamp: executor_machine_state.cycle_start_state.timestamp,
                            value: query.write_value,
                            timestamp_offset: query.local_timestamp_in_cycle as u32,
                        }
                    }
                };
                write_set.push(write_set_el);
            }

            let read_set_node = GrandProductAccumulationStep::Base {
                lhs: read_set[0].clone(),
                rhs: read_set[1].clone(),
                is_write: false,
            };
            graph.add_node(read_set_node.clone());
            grand_product_read_accumulation_nodes.push(read_set_node);

            let write_set_node = GrandProductAccumulationStep::Base {
                lhs: write_set[0].clone(),
                rhs: write_set[1].clone(),
                is_write: true,
            };
            graph.add_node(write_set_node.clone());
            grand_product_write_accumulation_nodes.push(write_set_node);
        }

        if shuffle_ram_augmented_sets.as_chunks::<2>().1.is_empty() == false {
            let last_el = shuffle_ram_augmented_sets.as_chunks::<2>().1[0].clone();
            // we tread PC permutation as a part of our global permutation under all the same rules

            let mut read_set = vec![];
            let mut write_set = vec![];

            // memory
            {
                let (query, aux) = last_el;
                let read_set_el = match query.query_type {
                    ShuffleRamQueryType::RegisterOnly { register_index } => {
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::SingleLimb(register_index),
                            address_space: AddressSpace::Constant(AddressSpaceType::Register),
                            timestamp: aux.read_timestamp,
                            value: query.read_value,
                            timestamp_offset: 0,
                        }
                    }
                    ShuffleRamQueryType::RegisterOrRam {
                        is_register,
                        address,
                    } => {
                        let address_space_inner = match is_register {
                            Boolean::Is(var) => AddressSpaceIsRegister::Is(var),
                            Boolean::Not(var) => AddressSpaceIsRegister::Not(var),
                            Boolean::Constant(..) => {
                                unreachable!()
                            }
                        };
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::U32Space(address),
                            address_space: AddressSpace::RegisterOrRam(address_space_inner),
                            timestamp: aux.read_timestamp,
                            value: query.read_value,
                            timestamp_offset: 0,
                        }
                    }
                };
                read_set.push(read_set_el);

                let write_set_el = match query.query_type {
                    ShuffleRamQueryType::RegisterOnly { register_index } => {
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::SingleLimb(register_index),
                            address_space: AddressSpace::Constant(AddressSpaceType::Register),
                            timestamp: executor_machine_state.cycle_start_state.timestamp,
                            value: query.write_value,
                            timestamp_offset: query.local_timestamp_in_cycle as u32,
                        }
                    }
                    ShuffleRamQueryType::RegisterOrRam {
                        is_register,
                        address,
                    } => {
                        let address_space_inner = match is_register {
                            Boolean::Is(var) => AddressSpaceIsRegister::Is(var),
                            Boolean::Not(var) => AddressSpaceIsRegister::Not(var),
                            Boolean::Constant(..) => {
                                unreachable!()
                            }
                        };
                        MemoryPermutationExpression {
                            address: AddressSpaceAddress::U32Space(address),
                            address_space: AddressSpace::RegisterOrRam(address_space_inner),
                            timestamp: executor_machine_state.cycle_start_state.timestamp,
                            value: query.write_value,
                            timestamp_offset: query.local_timestamp_in_cycle as u32,
                        }
                    }
                };
                write_set.push(write_set_el);
            }

            // and PC permutation
            {
                let read_set_el = MemoryPermutationExpression {
                    address: AddressSpaceAddress::Empty,
                    address_space: AddressSpace::Constant(AddressSpaceType::PC),
                    timestamp: executor_machine_state.cycle_start_state.timestamp,
                    value: executor_machine_state.cycle_start_state.pc,
                    timestamp_offset: 0,
                };
                read_set.push(read_set_el);

                let write_set_el = MemoryPermutationExpression {
                    address: AddressSpaceAddress::Empty,
                    address_space: AddressSpace::Constant(AddressSpaceType::PC),
                    timestamp: executor_machine_state.cycle_end_state.timestamp,
                    value: executor_machine_state.cycle_end_state.pc,
                    timestamp_offset: 0,
                };
                write_set.push(write_set_el);
            }

            let read_set_node = GrandProductAccumulationStep::Base {
                lhs: read_set[0].clone(),
                rhs: read_set[1].clone(),
                is_write: false,
            };
            graph.add_node(read_set_node.clone());
            grand_product_read_accumulation_nodes.push(read_set_node);

            let write_set_node = GrandProductAccumulationStep::Base {
                lhs: write_set[0].clone(),
                rhs: write_set[1].clone(),
                is_write: true,
            };
            graph.add_node(write_set_node.clone());
            grand_product_write_accumulation_nodes.push(write_set_node);
        } else {
            todo!();
        }

        // now we can follow up with lookup subarguments. We separate "hot" range check 16 and 19 bit
        // ones, and "generic" ones (that includes decoder)

        // Timestamps are the easiest - we collect them, and will transform into GKR layers very separately
        super::range_check_exprs::compile_timestamp_comparison_range_checks(
            &mut timestamp_range_check_expressions_to_compile,
            &shuffle_ram_augmented_sets,
            executor_machine_state.cycle_start_state.timestamp,
        );

        let total_timestamp_range_check_lookups =
            timestamp_range_check_expressions_to_compile.len() as u64 * trace_len as u64;
        assert!(total_timestamp_range_check_lookups < F::CHARACTERISTICS, "total number of timestamp range check lookups in circuit is {} that is larger that field characteristics {}", total_timestamp_range_check_lookups, F::CHARACTERISTICS);

        // Then 16-bit range checks
        let (range_check_16_expressions, mut generic_lookups) =
            super::range_check_exprs::split_range_check_exprs_from_compiler(
                &range_check_expressions,
            );

        let total_lookups_for_range_checks_16 =
            (range_check_16_expressions.len() as u64) * trace_len as u64;
        assert!(total_lookups_for_range_checks_16 < F::CHARACTERISTICS, "total number of range-check-16 lookups in circuit is {} that is larger that field characteristics {}", total_lookups_for_range_checks_16, F::CHARACTERISTICS);

        // and then all lookups from circuit + decoder are "generic" ones

        for lookup in lookups.iter() {
            let LookupQuery { row, table } = lookup.clone();
            generic_lookups.push((row.to_vec(), table));
        }
        // and decoder - we tightly pack it, and will need to do the same in the setup generator
        {
            // pub pc: [F; 2],
            // pub rs1_index: F,
            // pub rs2_index: F,
            // pub rd_index: F,
            // pub rd_is_zero: F,
            // pub imm: [F; REGISTER_SIZE],
            // pub funct3: F,
            // pub circuit_family_extra_mask: F,

            let mut decoder_lookup_trivial_vars = vec![];
            decoder_lookup_trivial_vars.extend(executor_machine_state.cycle_start_state.pc);
            decoder_lookup_trivial_vars.extend([executor_machine_state.decoder_data.rs1_index]);
            decoder_lookup_trivial_vars.extend([executor_machine_state.decoder_data.rs2_index]);
            decoder_lookup_trivial_vars.extend([executor_machine_state.decoder_data.rd_index]);
            decoder_lookup_trivial_vars.extend([executor_machine_state.decoder_data.rd_is_zero]);
            decoder_lookup_trivial_vars.extend(executor_machine_state.decoder_data.imm);
            if executor_machine_state.decoder_data.funct3.is_placeholder() == false {
                decoder_lookup_trivial_vars.extend([executor_machine_state.decoder_data.funct3]);
            }
            let mut decoder_lookup: Vec<_> = decoder_lookup_trivial_vars
                .into_iter()
                .map(|el| LookupInput::<F>::Variable(el))
                .collect();
            if circuit_family_bitmask.is_empty() {
                // nothing
            } else {
                if circuit_family_bitmask.len() == 1 {
                    // just variable itself
                    decoder_lookup.push(LookupInput::<F>::Variable(circuit_family_bitmask[0]));
                } else {
                    // constraint
                    assert!(circuit_family_bitmask.len() < F::CHAR_BITS);
                    let mut mask_constraint = Constraint::empty();
                    for (i, var) in circuit_family_bitmask.iter().enumerate() {
                        mask_constraint += Term::from((F::from_u64_unchecked(1 << i), *var));
                    }
                    mask_constraint.normalize();
                    let (_, linear_terms, _) = mask_constraint.split_max_quadratic();
                    decoder_lookup.push(LookupInput::Expression {
                        linear_terms,
                        constant_coeff: F::ZERO,
                    });
                }
            }
            generic_lookups.push((
                decoder_lookup,
                LookupQueryTableType::Constant(TableType::Decoder),
            ));
        }

        let mut max_width = generic_lookups
            .iter()
            .map(|el| el.0.len())
            .max()
            .unwrap_or(0);
        if max_width > 0 {
            max_width += 1;
        }
        println!("Generic lookup total tables in setup: {}", max_width);

        // for all boolean vars we add booleanity constraint here

        for boolean in boolean_vars.iter() {
            let t = Term::<F>::from(*boolean);
            let c = t.clone() * t.clone() - t;
            constraints.push((c, false));
        }

        // now we can optimize the constraints and all remaining variables
        for c in constraints.iter_mut() {
            c.0.normalize();
        }

        use crate::one_row_compiler::optimize_out_linear_constraints;
        let (optimized_out_variables, mut constraints) = optimize_out_linear_constraints(
            &state_input,
            &state_output,
            &substitutions,
            constraints,
            &mut all_variables_to_place,
        );

        println!(
            "{} variables were optimized out",
            optimized_out_variables.len()
        );

        for var in optimized_out_variables.iter() {
            if let Some(c) = variables_from_constraints.remove(var) {
                assert!(c.degree() < 2);
            }
        }

        for c in constraints.iter_mut() {
            c.0.normalize();
        }

        // and now we can finally layout all the variables. We do want to push all of them into intermediate layers

        // at the end of the day we are only left with:
        // - grand product accumulations
        // - lookup accumulations
        // - constraints

        // And we can transform them into (potentially intersecting) sub-GKRs

        // filter constraints that define variables
        let len_before = constraints.len();
        for (_, c) in variables_from_constraints.iter_mut() {
            c.normalize();
            let mut to_remove = None;
            for (idx, (cc, _)) in constraints.iter().enumerate() {
                if cc == &*c {
                    assert!(to_remove.is_none());
                    to_remove = Some(idx);
                }
            }
            if let Some(to_remove) = to_remove {
                constraints.remove(to_remove);
            }
        }
        let len_after = constraints.len();
        println!(
            "{} constraints removed as they define variables, will be used separately",
            len_before - len_after
        );

        // Above we only placed in the graph variables that have strict constraint on being at the base (input) layer of the proof.
        // Now we should try to move from the base layer and place the rest.
        // As we explicitly track variables that can be made "virtual" (by pushing them into intermediate GKR layers),
        // it should be relatively easy task.

        println!(
            "In total of {} variables are defined via constraints",
            variables_from_constraints.len()
        );
        for (var, _c) in variables_from_constraints.iter() {
            if let Some(name) = variable_names.get(var) {
                println!("Variable `{}` defined via constraint", name);
            }
        }

        // first define if any of the constraints depends on the variables defined via other constraints
        let mut variables_via_constraints_are_disjoint = true;
        let mut all_variables_in_constraints = HashSet::new();
        for (c, _) in constraints.iter() {
            for (var, _) in variables_from_constraints.iter() {
                if c.contains_var(var) {
                    variables_via_constraints_are_disjoint = false;
                }
            }
            c.dump_variables(&mut all_variables_in_constraints);
        }

        println!(
            "Variables defined via constraints are disjoint = {}",
            variables_via_constraints_are_disjoint
        );

        // now we will make small heuristic decision to verifier tradeoff - if we have too little of constraint defined vars,
        // or if adding them to witness cost nothing for prover - we will instead push them to witness

        const MIN_VARS_VIA_CONSTRAINTS: usize = 8;
        const HASHING_ROUND_SIZE: usize = 16;

        let vars_to_place = all_variables_to_place.len();
        let vars_to_place_if_using_constraints = vars_to_place - variables_from_constraints.len();
        let rounds = vars_to_place.div_ceil(HASHING_ROUND_SIZE);
        let rounds_if_using_constraints =
            vars_to_place_if_using_constraints.div_ceil(HASHING_ROUND_SIZE);

        let push_via_constraints_into_witness = if variables_from_constraints.len()
            < MIN_VARS_VIA_CONSTRAINTS
            || rounds == rounds_if_using_constraints
        {
            true
        } else {
            false
        };

        if push_via_constraints_into_witness {
            // place all variables to place into witness, and push constraints back
            for (v, c) in variables_from_constraints.into_iter() {
                assert!(all_variables_to_place.contains(&v));
                constraints.push((c, true));
            }

            // put all variables into base layer
            for var in all_variables_to_place.clone().iter() {
                let _ = graph
                    .layout_witness_subtree_multiple_variables([*var], &mut all_variables_to_place);
            }
            assert!(all_variables_to_place.is_empty());
        } else {
            todo!();
        }

        // Accumulate grand product - pairwise as much as we can

        let (final_read_node, final_write_node) = {
            let mut next_read_set = vec![];
            let mut next_write_set = vec![];

            for [a, b] in grand_product_read_accumulation_nodes
                .as_chunks::<2>()
                .0
                .iter()
            {
                let el = GrandProductAccumulationStep::Aggregation {
                    lhs: Box::new(a.clone()),
                    rhs: Box::new(b.clone()),
                    is_write: false,
                };
                graph.add_node(el.clone());
                next_read_set.push(el);
            }

            for [a, b] in grand_product_write_accumulation_nodes
                .as_chunks::<2>()
                .0
                .iter()
            {
                let el = GrandProductAccumulationStep::Aggregation {
                    lhs: Box::new(a.clone()),
                    rhs: Box::new(b.clone()),
                    is_write: true,
                };
                graph.add_node(el.clone());
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
                current_read_remainder =
                    Some(grand_product_read_accumulation_nodes.as_chunks::<2>().1[0].clone());
            }
            if grand_product_write_accumulation_nodes
                .as_chunks::<2>()
                .1
                .len()
                > 0
            {
                current_write_remainder =
                    Some(grand_product_write_accumulation_nodes.as_chunks::<2>().1[0].clone());
            }

            let mut next_read_set = vec![];
            let mut next_write_set = vec![];
            let mut next_read_remainder = None;
            let mut next_write_remainder = None;

            // if base_read_remainder.is_some()
            //     || base_write_remainder.is_some()
            if current_read_set.len() > 1 || current_write_set.len() > 1 {
                loop {
                    for [a, b] in current_read_set.as_chunks::<2>().0.iter() {
                        let el = GrandProductAccumulationStep::Aggregation {
                            lhs: Box::new(a.clone()),
                            rhs: Box::new(b.clone()),
                            is_write: false,
                        };
                        graph.add_node(el.clone());
                        next_read_set.push(el);
                    }

                    for [a, b] in current_write_set.as_chunks::<2>().0.iter() {
                        let el = GrandProductAccumulationStep::Aggregation {
                            lhs: Box::new(a.clone()),
                            rhs: Box::new(b.clone()),
                            is_write: true,
                        };
                        graph.add_node(el.clone());
                        next_write_set.push(el);
                    }

                    if current_read_set.as_chunks::<2>().1.len() > 0 {
                        if let Some(current_read_remainder) = current_read_remainder.take() {
                            let el = GrandProductAccumulationStep::Aggregation {
                                lhs: Box::new(current_read_set.as_chunks::<2>().1[0].clone()),
                                rhs: Box::new(current_read_remainder),
                                is_write: false,
                            };
                            graph.add_node(el.clone());
                            next_read_set.push(el);
                        } else {
                            next_read_remainder =
                                Some(current_read_set.as_chunks::<2>().1[0].clone());
                        }
                    }

                    if current_write_set.as_chunks::<2>().1.len() > 0 {
                        if let Some(current_write_remainder) = current_write_remainder.take() {
                            let el = GrandProductAccumulationStep::Aggregation {
                                lhs: Box::new(current_write_set.as_chunks::<2>().1[0].clone()),
                                rhs: Box::new(current_write_remainder),
                                is_write: false,
                            };
                            graph.add_node(el.clone());
                            next_write_set.push(el);
                        } else {
                            next_write_remainder =
                                Some(current_write_set.as_chunks::<2>().1[0].clone());
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

            let read_mask = GrandProductAccumulationMaskingNode {
                lhs: read_node,
                mask: graph
                    .get_fixed_layout_pos(&executor_machine_state.execute)
                    .expect("already placed"),
                is_write: false,
            };
            graph.add_node(read_mask.clone());

            let write_mask = GrandProductAccumulationMaskingNode {
                lhs: write_node,
                mask: graph
                    .get_fixed_layout_pos(&executor_machine_state.execute)
                    .expect("already placed"),
                is_write: true,
            };
            graph.add_node(write_mask.clone());

            (read_mask, write_mask)
        };

        // placing lookup is move involved
        let (final_lookup_numerator_nodes, final_lookup_denominator_nodes) = {
            if range_check_16_expressions.len() > 0 {
                layout_width_1_lookup_expressions(
                    &mut graph,
                    range_check_16_expressions,
                    &mut num_variables,
                    &mut all_variables_to_place,
                    &mut variable_names,
                    "range check 16",
                    LookupType::RangeCheck16,
                );
            }

            if timestamp_range_check_expressions_to_compile.len() > 0 {
                layout_width_1_lookup_expressions(
                    &mut graph,
                    timestamp_range_check_expressions_to_compile,
                    &mut num_variables,
                    &mut all_variables_to_place,
                    &mut variable_names,
                    "timestamp range check",
                    LookupType::TimestampRangeCheck,
                );
            }

            if generic_lookups.len() > 0 {
                let generic_lookups = generic_lookups
                    .into_iter()
                    .map(|el| (el.0, LookupQueryTableTypeExt::from_simple(el.1)))
                    .collect();
                layout_lookup_expressions::<F, 10>(
                    &mut graph,
                    generic_lookups,
                    &mut num_variables,
                    &mut all_variables_to_place,
                    &mut variable_names,
                    "generic lookup",
                    Some(executor_machine_state.execute),
                    LookupType::Generic,
                );
            }

            ((), ())
        };

        // Dealing with constraints is simple - we will perform two step reduction:
        // - first all quadratic parts from all constraints are delinearized and summed
        // - then we compute execute * (quadratic + \sum linears + \sum constants

        let _ = layout_constraints(&mut graph, constraints, executor_machine_state.execute);

        let graphviz_string = graph.make_graphviz(&variable_names);
        println!("{}", &graphviz_string);
        let gv_graph = ::layout::gv::DotParser::new(&graphviz_string)
            .process()
            .unwrap();
        let mut builder = ::layout::gv::GraphBuilder::new();
        builder.visit_graph(&gv_graph);
        let mut svg = ::layout::backends::svg::SVGWriter::new();
        let mut vg = builder.get();
        vg.do_it(false, false, false, &mut svg);
        let content = svg.finalize();
        let filename = "gkr_layout.svg";
        ::layout::core::utils::save_to_file(filename, &content).unwrap();

        let _ = graph.layout_layers();

        todo!();
    }
}
