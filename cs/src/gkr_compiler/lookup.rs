use crate::cs::circuit::LookupQueryTableTypeExt;
use crate::definitions::gkr::DECODER_LOOKUP_FORMAL_SET_INDEX;
use crate::definitions::{Degree1Constraint, GKRAddress, Variable};
use crate::gkr_compiler::graph::{GKRGraph, GraphHolder};
use crate::gkr_compiler::lookup_nodes::LookupInputRelation;
use crate::gkr_compiler::lookup_nodes::LookupRationalPair;
use crate::one_row_compiler::LookupInput;
use crate::tables::TableType;

use super::compiled_constraint::GKRCompiledLinearConstraint;
use super::*;

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
    LookupRationalPair,
    NoFieldGKRRelation,
    Vec<NoFieldSingleColumnLookupRelation>,
) {
    let (a, b, c, rels) = layout_lookup_expressions::<F, true>(
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
        1,
        false,
    );

    let rels = rels
        .into_iter()
        .map(|el| {
            assert_eq!(el.columns.len(), 1);

            NoFieldSingleColumnLookupRelation {
                input: el.columns[0].clone(),
                lookup_set_index: el.lookup_set_index,
            }
        })
        .collect();

    (a, b, c, rels)
}

fn lookup_input_node_from_expr<F: PrimeField, const SINGLE_COLUMN: bool>(
    expr: &(Vec<LookupInput<F>>, LookupQueryTableTypeExt<F>),
    total_width: usize,
    expect_table_id: bool,
) -> LookupInputRelation<F> {
    let (expr, table_type) = expr;
    if SINGLE_COLUMN {
        assert!(expect_table_id == false);
        assert_eq!(total_width, 1);
        assert_eq!(expr.len(), 1);
        assert_eq!(
            *table_type,
            LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
        );
    } else {
        assert!(expr.len() <= total_width)
    }

    let mut inputs = Vec::new();

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
    if SINGLE_COLUMN == false && expect_table_id {
        assert_ne!(
            *table_type,
            LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
        );
        match table_type {
            LookupQueryTableTypeExt::Constant(constant) => {
                inputs.push(Degree1Constraint {
                    linear_terms: vec![].into_boxed_slice(),
                    constant_term: F::from_u32_unchecked(constant.to_table_id() as u32),
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

    assert!(
        inputs.len() <= total_width,
        "expression {:?} was compiled into {} terms, but expected total width is {}",
        expr,
        inputs.len(),
        total_width
    );

    LookupInputRelation { inputs }
}

pub(crate) fn layout_lookup_expressions<F: PrimeField, const SINGLE_COLUMN: bool>(
    graph: &mut GKRGraph,
    expressions: Vec<(Vec<LookupInput<F>>, LookupQueryTableTypeExt<F>)>,
    num_variables: &mut u64,
    all_variables_to_place: &mut BTreeSet<Variable>,
    variable_names: &mut HashMap<Variable, String>,
    lookup_type: &str,
    decoder_lookup: Option<(Variable, Vec<LookupInput<F>>)>,
    lookup: LookupType,
    total_width: usize,
    expect_table_id: bool,
) -> (
    Variable,
    LookupRationalPair,
    NoFieldGKRRelation,
    Vec<NoFieldVectorLookupRelation>,
) {
    let mut initial_relations = vec![];

    for (idx, rel) in expressions.iter().enumerate() {
        let input =
            lookup_input_node_from_expr::<F, SINGLE_COLUMN>(rel, total_width, expect_table_id);
        let input = lookup_input_into_relation::<F, SINGLE_COLUMN>(&input, idx, &*graph);
        initial_relations.push(input);
    }
    // if let Some(decoder_lookup) = decoder_lookup {
    //     let (decoder_predicate_var, decoder_lookup) = decoder_lookup;
    //     let decoder_predicate = graph.get_address_for_variable(decoder_predicate_var);
    //     let input = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
    //         &(
    //             decoder_lookup,
    //             LookupQueryTableTypeExt::Constant(TableType::Decoder),
    //         ),
    //         total_width,
    //         expect_table_id,
    //     );
    //     let input = lookup_input_into_relation::<F, SINGLE_COLUMN>(&input, &*graph);
    //     assert_eq!(input.0.len(), graph.setup_addresses(lookup).len());
    //     initial_relations.push(input);
    // }

    println!(
        "In total of {} lookups of type {}",
        expressions.len(),
        lookup_type
    );
    if decoder_lookup.is_some() {
        println!("Decoder lookup is present");
    }

    let single_columns_lookup_width = match lookup {
        LookupType::RangeCheck16 => Some(16),
        LookupType::TimestampRangeCheck => Some(TIMESTAMP_COLUMNS_NUM_BITS),
        LookupType::Generic => None,
    };

    // create multiplicity
    let multiplicity_var = Variable(*num_variables);
    variable_names.insert(
        multiplicity_var,
        format!("Multiplicity for {}", lookup_type),
    );
    *num_variables += 1;
    all_variables_to_place.insert(multiplicity_var);
    let [multiplicity_pos] =
        graph.layout_witness_subtree_multiple_variables([multiplicity_var], all_variables_to_place);

    for (expr, table_type) in expressions.iter() {
        if SINGLE_COLUMN {
            assert_eq!(total_width, 1);
            assert_eq!(expr.len(), 1);
            assert_eq!(
                *table_type,
                LookupQueryTableTypeExt::Constant(TableType::DynamicPlaceholder)
            );
        } else {
            assert!(expr.len() <= total_width);
        }
    }

    let total_rational_terms = expressions.len() + 1 + (decoder_lookup.is_some() as usize);

    let mut initial_reduction_layer_nodes = vec![];
    assert!(total_rational_terms > 0);
    let mut placement_layer = 1;

    if total_rational_terms % 2 == 0 {
        if let Some(decoder_lookup) = decoder_lookup {
            {
                let (decoder_predicate_var, decoder_lookup) = decoder_lookup;
                let decoder_predicate = graph.get_address_for_variable(decoder_predicate_var);
                let input = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &(
                        decoder_lookup,
                        LookupQueryTableTypeExt::Constant(TableType::Decoder),
                    ),
                    total_width,
                    expect_table_id,
                );
                let input = lookup_input_into_relation::<F, SINGLE_COLUMN>(
                    &input,
                    DECODER_LOOKUP_FORMAL_SET_INDEX,
                    &*graph,
                );
                assert_eq!(input.columns.len(), graph.setup_addresses(lookup).len());

                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Positive(decoder_predicate),
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<SINGLE_COLUMN>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                if expressions.is_empty() {
                    return (multiplicity_var, next_pair, rel, initial_relations);
                }

                initial_reduction_layer_nodes.push((next_pair, rel));
            }

            // and continue over all other pairs
            assert_eq!(expressions.len() % 2, 0);
            let mut set_idx = 0;
            for [a, b] in expressions.as_chunks::<2>().0 {
                // We will take 2 inputs with "1" in the numerator, and some witness in denominator
                let a = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &a,
                    total_width,
                    expect_table_id,
                );
                let b = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &b,
                    total_width,
                    expect_table_id,
                );
                let a = lookup_input_into_relation::<F, SINGLE_COLUMN>(&a, set_idx, &*graph);
                set_idx += 1;
                let b = lookup_input_into_relation::<F, SINGLE_COLUMN>(&b, set_idx, &*graph);
                set_idx += 1;
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                initial_reduction_layer_nodes.push((next_pair, rel));
            }
        } else {
            // we will make a mixed node with one of the witnesses to avoid copying multiplicity
            assert_eq!(expressions.len() % 2, 1);
            let (first, expressions) = expressions.split_at(1);
            let first = &first[0];
            let mut set_idx = 0;
            {
                let input = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    first,
                    total_width,
                    expect_table_id,
                );
                let input =
                    lookup_input_into_relation::<F, SINGLE_COLUMN>(&input, set_idx, &*graph);
                set_idx += 1;
                assert_eq!(input.columns.len(), graph.setup_addresses(lookup).len());

                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<SINGLE_COLUMN>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                if expressions.is_empty() {
                    return (multiplicity_var, next_pair, rel, initial_relations);
                }

                initial_reduction_layer_nodes.push((next_pair, rel));
            }

            assert_eq!(expressions.len() % 2, 0);
            for [a, b] in expressions.as_chunks::<2>().0 {
                let a = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &a,
                    total_width,
                    expect_table_id,
                );
                let b = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &b,
                    total_width,
                    expect_table_id,
                );
                let a = lookup_input_into_relation::<F, SINGLE_COLUMN>(&a, set_idx, &*graph);
                set_idx += 1;
                let b = lookup_input_into_relation::<F, SINGLE_COLUMN>(&b, set_idx, &*graph);
                set_idx += 1;
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                initial_reduction_layer_nodes.push((next_pair, rel));
            }
        }
    } else {
        // inevitably we will need to copy something, so we will try to copy the simplest case
        if let Some(decoder_lookup) = decoder_lookup {
            assert!(expressions.is_empty() == false);
            {
                let (decoder_predicate_var, decoder_lookup) = decoder_lookup;
                let decoder_predicate = graph.get_address_for_variable(decoder_predicate_var);
                let input = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &(
                        decoder_lookup,
                        LookupQueryTableTypeExt::Constant(TableType::Decoder),
                    ),
                    total_width,
                    expect_table_id,
                );
                let input = lookup_input_into_relation::<F, SINGLE_COLUMN>(
                    &input,
                    DECODER_LOOKUP_FORMAL_SET_INDEX,
                    &*graph,
                );
                assert_eq!(input.columns.len(), graph.setup_addresses(lookup).len());

                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Positive(decoder_predicate),
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<SINGLE_COLUMN>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                if expressions.is_empty() {
                    return (multiplicity_var, next_pair, rel, initial_relations);
                }

                initial_reduction_layer_nodes.push((next_pair, rel));
            }

            // and continue over all other pairs
            assert_eq!(expressions.len() % 2, 1);
            let mut set_idx = 0;
            for [a, b] in expressions.as_chunks::<2>().0 {
                let a = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &a,
                    total_width,
                    expect_table_id,
                );
                let b = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &b,
                    total_width,
                    expect_table_id,
                );
                let a = lookup_input_into_relation::<F, SINGLE_COLUMN>(&a, set_idx, &*graph);
                set_idx += 1;
                let b = lookup_input_into_relation::<F, SINGLE_COLUMN>(&b, set_idx, &*graph);
                set_idx += 1;
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                initial_reduction_layer_nodes.push((next_pair, rel));
            }
            {
                let last_input = expressions.as_chunks::<2>().1[0].clone();
                let last_input = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &last_input,
                    total_width,
                    expect_table_id,
                );
                let last_input =
                    lookup_input_into_relation::<F, SINGLE_COLUMN>(&last_input, set_idx, &*graph);
                let last_input = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: copy_single_base_input_or_materialize_vector::<SINGLE_COLUMN>(last_input),
                    den_node: None,
                    lookup_type: lookup,
                };

                let (next_pair, rel) = LookupRationalPair::add_single_into_graph(
                    last_input,
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                initial_reduction_layer_nodes.push((next_pair, rel));
            }
        } else {
            // we will make a mixed node with one of the witnesses to avoid copying multiplicity and setup
            assert_eq!(expressions.len() % 2, 0);
            let (first, expressions) = expressions.split_at(1);
            let first = &first[0];
            let mut set_idx = 0;
            {
                let input = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    first,
                    total_width,
                    expect_table_id,
                );
                let input =
                    lookup_input_into_relation::<F, SINGLE_COLUMN>(&input, set_idx, &*graph);
                set_idx += 1;
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(input),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Negative(multiplicity_pos),
                    num_node: None,
                    den: vector_or_single_setup::<SINGLE_COLUMN>(graph, lookup),
                    den_node: None,
                    lookup_type: lookup,
                };

                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                initial_reduction_layer_nodes.push((next_pair, rel));
            }

            assert_eq!(expressions.len() % 2, 1);
            for [a, b] in expressions.as_chunks::<2>().0 {
                let a = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &a,
                    total_width,
                    expect_table_id,
                );
                let b = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &b,
                    total_width,
                    expect_table_id,
                );
                let a = lookup_input_into_relation::<F, SINGLE_COLUMN>(&a, set_idx, &*graph);
                set_idx += 1;
                let b = lookup_input_into_relation::<F, SINGLE_COLUMN>(&b, set_idx, &*graph);
                set_idx += 1;
                let a = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(a),
                    den_node: None,
                    lookup_type: lookup,
                };
                let b = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: vector_or_single_input::<SINGLE_COLUMN>(b),
                    den_node: None,
                    lookup_type: lookup,
                };
                let (next_pair, rel) = LookupRationalPair::accumulate_pair_into_graph(
                    (a, b),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                initial_reduction_layer_nodes.push((next_pair, rel));
            }
            {
                let last_input = expressions.as_chunks::<2>().1[0].clone();
                let last_input = lookup_input_node_from_expr::<F, SINGLE_COLUMN>(
                    &last_input,
                    total_width,
                    expect_table_id,
                );
                let last_input =
                    lookup_input_into_relation::<F, SINGLE_COLUMN>(&last_input, set_idx, &*graph);
                let last_input = LookupRationalPair {
                    num: lookup_nodes::LookupNumerator::Identity,
                    num_node: None,
                    den: copy_single_base_input_or_materialize_vector::<SINGLE_COLUMN>(last_input),
                    den_node: None,
                    lookup_type: lookup,
                };

                let (next_pair, rel) = LookupRationalPair::add_single_into_graph(
                    last_input,
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                initial_reduction_layer_nodes.push((next_pair, rel));
            }
        }
    }

    // now we resolved a problem of copying from base layer, but we still want to have all the relations to be between two
    // nearby layers only

    println!(
        "Will continue placement of {} lookup rationals into layer {}",
        initial_reduction_layer_nodes.len(),
        placement_layer + 1
    );

    let mut current_layer = initial_reduction_layer_nodes;

    loop {
        if current_layer.len() == 1 {
            let (last_pair, rel) = current_layer.pop().unwrap();
            return (multiplicity_var, last_pair, rel, initial_relations);
        }

        placement_layer += 1;
        let mut next_layer = vec![];
        for [a, b] in current_layer.as_chunks::<2>().0.iter() {
            let next_pair = LookupRationalPair::accumulate_pair_into_graph(
                (a.0.clone(), b.0.clone()),
                graph,
                placement_layer,
                single_columns_lookup_width,
            );

            next_layer.push(next_pair);
        }
        match current_layer.as_chunks::<2>().1 {
            [] => {}
            [last] => {
                let next_pair = LookupRationalPair::add_single_into_graph(
                    last.0.clone(),
                    graph,
                    placement_layer,
                    single_columns_lookup_width,
                );

                next_layer.push(next_pair);
            }
            _ => {
                unreachable!()
            }
        }

        current_layer = next_layer;
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
