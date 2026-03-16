use proc_macro2::TokenStream;
use quote::{quote, TokenStreamExt};

use prover::cs::definitions::GKRAddress;
use prover::cs::gkr_compiler::{
    GKRCircuitArtifact, GKRLayerDescription, GateArtifacts, NoFieldGKRRelation,
    NoFieldMaxQuadraticConstraintsGKRRelation, OutputType,
};
use prover::cs::one_row_compiler::gkr::{
    NoFieldLinearRelation, NoFieldSingleColumnLookupRelation, NoFieldVectorLookupRelation,
};
use prover::field::baby_bear::base::BabyBearField;
use prover::field::{Field, FieldExtension, PrimeField};
use prover::gkr::prover::{GKRExternalChallenges, GKRProof};
use prover::merkle_trees::ColumnMajorMerkleTreeConstructor;

fn coeff_to_montgomery(coeff: u32) -> u32 {
    BabyBearField::from_nonreduced_u32(coeff).raw_u32_value()
}

// --- TokenStream builders for GKR address and type enums ---

fn transform_gkr_address(addr: &GKRAddress) -> TokenStream {
    match addr {
        GKRAddress::BaseLayerWitness(offset) => {
            quote! { GKRAddress::BaseLayerWitness(#offset) }
        }
        GKRAddress::BaseLayerMemory(offset) => {
            quote! { GKRAddress::BaseLayerMemory(#offset) }
        }
        GKRAddress::InnerLayer { layer, offset } => {
            quote! { GKRAddress::InnerLayer { layer: #layer, offset: #offset } }
        }
        GKRAddress::Setup(offset) => {
            quote! { GKRAddress::Setup(#offset) }
        }
        GKRAddress::OptimizedOut(offset) => {
            quote! { GKRAddress::OptimizedOut(#offset) }
        }
        GKRAddress::Cached { layer, offset } => {
            quote! { GKRAddress::Cached { layer: #layer, offset: #offset } }
        }
    }
}

fn transform_output_type(ot: &OutputType) -> TokenStream {
    match ot {
        OutputType::PermutationProduct => quote! { OutputType::PermutationProduct },
        OutputType::Lookup16Bits => quote! { OutputType::Lookup16Bits },
        OutputType::LookupTimestamps => quote! { OutputType::LookupTimestamps },
        OutputType::GenericLookup => quote! { OutputType::GenericLookup },
    }
}

fn addr_to_idx(addr: &GKRAddress, sorted_addrs: &[GKRAddress]) -> usize {
    sorted_addrs
        .binary_search(addr)
        .unwrap_or_else(|_| panic!("address {:?} not found in sorted addrs", addr))
}

fn transform_linear_relation(
    rel: &NoFieldLinearRelation,
    input_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    let constant = rel.constant;
    let mut terms_stream = TokenStream::new();
    terms_stream.append_separated(
        rel.linear_terms.iter().map(|(coeff, addr)| {
            let idx = addr_to_idx(addr, input_sorted_addrs);
            quote! { (#coeff, #idx) }
        }),
        quote! {,},
    );
    quote! {
        StaticNoFieldLinearRelation {
            linear_terms: &[#terms_stream],
            constant: #constant,
        }
    }
}

fn transform_single_column_lookup(
    rel: &NoFieldSingleColumnLookupRelation,
    input_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    let input = transform_linear_relation(&rel.input, input_sorted_addrs);
    let lookup_set_index = rel.lookup_set_index;
    quote! {
        StaticNoFieldSingleColumnLookupRelation {
            input: #input,
            lookup_set_index: #lookup_set_index,
        }
    }
}

fn transform_vector_lookup(
    rel: &NoFieldVectorLookupRelation,
    input_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    let mut columns_stream = TokenStream::new();
    columns_stream.append_separated(
        rel.columns
            .iter()
            .map(|c| transform_linear_relation(c, input_sorted_addrs)),
        quote! {,},
    );
    let lookup_set_index = rel.lookup_set_index;
    quote! {
        StaticNoFieldVectorLookupRelation {
            columns: &[#columns_stream],
            lookup_set_index: #lookup_set_index,
        }
    }
}

fn transform_max_quadratic_relation(
    rel: &NoFieldMaxQuadraticConstraintsGKRRelation,
    input_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    let mut qt_stream = TokenStream::new();
    qt_stream.append_separated(
        rel.quadratic_terms.iter().map(|((a, b), terms)| {
            let a_idx = addr_to_idx(a, input_sorted_addrs);
            let b_idx = addr_to_idx(b, input_sorted_addrs);
            let mut inner = TokenStream::new();
            inner.append_separated(
                terms.iter().map(|(coeff, pow)| {
                    let mont = coeff_to_montgomery(*coeff);
                    quote! { (#mont, #pow) }
                }),
                quote! {,},
            );
            quote! { ((#a_idx, #b_idx), &[#inner] as &[(u32, usize)]) }
        }),
        quote! {,},
    );

    let mut lt_stream = TokenStream::new();
    lt_stream.append_separated(
        rel.linear_terms.iter().map(|(addr, terms)| {
            let idx = addr_to_idx(addr, input_sorted_addrs);
            let mut inner = TokenStream::new();
            inner.append_separated(
                terms.iter().map(|(coeff, pow)| {
                    let mont = coeff_to_montgomery(*coeff);
                    quote! { (#mont, #pow) }
                }),
                quote! {,},
            );
            quote! { (#idx, &[#inner] as &[(u32, usize)]) }
        }),
        quote! {,},
    );

    let mut ct_stream = TokenStream::new();
    ct_stream.append_separated(
        rel.constants.iter().map(|(coeff, pow)| {
            let mont = coeff_to_montgomery(*coeff);
            quote! { (#mont, #pow) }
        }),
        quote! {,},
    );

    quote! {
        StaticNoFieldMaxQuadraticConstraintsGKRRelation {
            quadratic_terms: &[#qt_stream],
            linear_terms: &[#lt_stream],
            constants: &[#ct_stream],
        }
    }
}

fn transform_relation(
    rel: &NoFieldGKRRelation,
    input_sorted_addrs: &[GKRAddress],
    output_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    use NoFieldGKRRelation as R;
    let in_idx = |addr: &GKRAddress| addr_to_idx(addr, input_sorted_addrs);
    let out_idx = |addr: &GKRAddress| addr_to_idx(addr, output_sorted_addrs);

    match rel {
        R::EnforceConstraintsMaxQuadratic { input } => {
            let input = transform_max_quadratic_relation(input, input_sorted_addrs);
            quote! {
                StaticNoFieldGKRRelation::EnforceConstraintsMaxQuadratic { input: #input }
            }
        }
        R::Copy { input, output } => {
            let i = in_idx(input);
            let o = out_idx(output);
            quote! { StaticNoFieldGKRRelation::Copy { input: #i, output: #o } }
        }
        R::InitialGrandProductFromCaches { input, output } => {
            let i0 = in_idx(&input[0]);
            let i1 = in_idx(&input[1]);
            let o = out_idx(output);
            quote! { StaticNoFieldGKRRelation::InitialGrandProductFromCaches { input: [#i0, #i1], output: #o } }
        }
        R::UnbalancedGrandProductWithCache {
            scalar,
            input,
            output,
        } => {
            let s = in_idx(scalar);
            let i = in_idx(input);
            let o = out_idx(output);
            quote! { StaticNoFieldGKRRelation::UnbalancedGrandProductWithCache { scalar: #s, input: #i, output: #o } }
        }
        R::TrivialProduct { input, output } => {
            let i0 = in_idx(&input[0]);
            let i1 = in_idx(&input[1]);
            let o = out_idx(output);
            quote! { StaticNoFieldGKRRelation::TrivialProduct { input: [#i0, #i1], output: #o } }
        }
        R::MaskIntoIdentityProduct {
            input,
            mask,
            output,
        } => {
            let i = in_idx(input);
            let m = in_idx(mask);
            let o = out_idx(output);
            quote! { StaticNoFieldGKRRelation::MaskIntoIdentityProduct { input: #i, mask: #m, output: #o } }
        }
        R::MaterializeSingleLookupInput { input, output } => {
            let i = transform_single_column_lookup(input, input_sorted_addrs);
            let o = out_idx(output);
            quote! { StaticNoFieldGKRRelation::MaterializeSingleLookupInput { input: #i, output: #o } }
        }
        R::MaterializedVectorLookupInput { input, output } => {
            let i = transform_vector_lookup(input, input_sorted_addrs);
            let o = out_idx(output);
            quote! { StaticNoFieldGKRRelation::MaterializedVectorLookupInput { input: #i, output: #o } }
        }
        R::LookupWithCachedDensAndSetup {
            input,
            setup,
            output,
        } => {
            let i0 = in_idx(&input[0]);
            let i1 = in_idx(&input[1]);
            let s0 = in_idx(&setup[0]);
            let s1 = in_idx(&setup[1]);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupWithCachedDensAndSetup { input: [#i0, #i1], setup: [#s0, #s1], output: [#o0, #o1] } }
        }
        R::LookupPairFromBaseInputs { input, output } => {
            let i0 = transform_single_column_lookup(&input[0], input_sorted_addrs);
            let i1 = transform_single_column_lookup(&input[1], input_sorted_addrs);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPairFromBaseInputs { input: [#i0, #i1], output: [#o0, #o1] } }
        }
        R::LookupPairFromMaterializedBaseInputs { input, output } => {
            let i0 = in_idx(&input[0]);
            let i1 = in_idx(&input[1]);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPairFromMaterializedBaseInputs { input: [#i0, #i1], output: [#o0, #o1] } }
        }
        R::LookupUnbalancedPairWithBaseInputs {
            input,
            remainder,
            output,
        } => {
            let i0 = in_idx(&input[0]);
            let i1 = in_idx(&input[1]);
            let r = transform_single_column_lookup(remainder, input_sorted_addrs);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupUnbalancedPairWithBaseInputs { input: [#i0, #i1], remainder: #r, output: [#o0, #o1] } }
        }
        R::LookupFromBaseInputsWithSetup {
            input,
            setup,
            output,
        } => {
            let i = transform_single_column_lookup(input, input_sorted_addrs);
            let s0 = in_idx(&setup[0]);
            let s1 = in_idx(&setup[1]);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupFromBaseInputsWithSetup { input: #i, setup: [#s0, #s1], output: [#o0, #o1] } }
        }
        R::LookupFromMaterializedBaseInputWithSetup {
            input,
            setup,
            output,
        } => {
            let i = in_idx(input);
            let s0 = in_idx(&setup[0]);
            let s1 = in_idx(&setup[1]);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupFromMaterializedBaseInputWithSetup { input: #i, setup: [#s0, #s1], output: [#o0, #o1] } }
        }
        R::LookupUnbalancedPairWithMaterializedBaseInputs {
            input,
            remainder,
            output,
        } => {
            let i0 = in_idx(&input[0]);
            let i1 = in_idx(&input[1]);
            let r = in_idx(remainder);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupUnbalancedPairWithMaterializedBaseInputs { input: [#i0, #i1], remainder: #r, output: [#o0, #o1] } }
        }
        R::LookupPairFromVectorInputs { input, output } => {
            let i0 = transform_vector_lookup(&input[0], input_sorted_addrs);
            let i1 = transform_vector_lookup(&input[1], input_sorted_addrs);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPairFromVectorInputs { input: [#i0, #i1], output: [#o0, #o1] } }
        }
        R::LookupPair { input, output } => {
            let i00 = in_idx(&input[0][0]);
            let i01 = in_idx(&input[0][1]);
            let i10 = in_idx(&input[1][0]);
            let i11 = in_idx(&input[1][1]);
            let o0 = out_idx(&output[0]);
            let o1 = out_idx(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPair { input: [[#i00, #i01], [#i10, #i11]], output: [#o0, #o1] } }
        }
    }
}

fn transform_gate(
    gate: &GateArtifacts,
    input_sorted_addrs: &[GKRAddress],
    output_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    let output_layer = gate.output_layer;
    let rel = transform_relation(&gate.enforced_relation, input_sorted_addrs, output_sorted_addrs);
    quote! {
        StaticGateArtifacts {
            output_layer: #output_layer,
            enforced_relation: #rel,
        }
    }
}

fn transform_layer_description(
    layer: &GKRLayerDescription,
    layer_idx: usize,
    input_sorted_addrs: &[GKRAddress],
    output_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    let gates_name = quote::format_ident!("LAYER_{}_GATES", layer_idx);
    let ext_gates_name = quote::format_ident!("LAYER_{}_EXT_GATES", layer_idx);
    let base_openings_name = quote::format_ident!("LAYER_{}_BASE_OPENINGS", layer_idx);
    let desc_name = quote::format_ident!("LAYER_{}_DESC", layer_idx);

    let mut gates_stream = TokenStream::new();
    gates_stream.append_separated(
        layer
            .gates
            .iter()
            .map(|g| transform_gate(g, input_sorted_addrs, output_sorted_addrs)),
        quote! {,},
    );

    let mut ext_gates_stream = TokenStream::new();
    ext_gates_stream.append_separated(
        layer
            .gates_with_external_connections
            .iter()
            .map(|g| transform_gate(g, input_sorted_addrs, output_sorted_addrs)),
        quote! {,},
    );

    let mut base_openings_stream = TokenStream::new();
    base_openings_stream.append_separated(
        layer
            .additional_base_layer_openings
            .iter()
            .map(|a| transform_gkr_address(a)),
        quote! {,},
    );

    quote! {
        const #gates_name: &[StaticGateArtifacts<'static>] = &[#gates_stream];
        const #ext_gates_name: &[StaticGateArtifacts<'static>] = &[#ext_gates_stream];
        const #base_openings_name: &[GKRAddress] = &[#base_openings_stream];
        const #desc_name: StaticGKRLayerDescription<'static> = StaticGKRLayerDescription {
            gates: #gates_name,
            gates_with_external_connections: #ext_gates_name,
            additional_base_layer_openings: #base_openings_name,
        };
    }
}

fn collect_addrs_from_linear_relation(
    rel: &NoFieldLinearRelation,
    addrs: &mut std::collections::BTreeSet<GKRAddress>,
) {
    for (_, addr) in &rel.linear_terms {
        addrs.insert(*addr);
    }
}

fn collect_addrs_from_single_lookup(
    rel: &NoFieldSingleColumnLookupRelation,
    addrs: &mut std::collections::BTreeSet<GKRAddress>,
) {
    collect_addrs_from_linear_relation(&rel.input, addrs);
}

fn collect_addrs_from_vector_lookup(
    rel: &NoFieldVectorLookupRelation,
    addrs: &mut std::collections::BTreeSet<GKRAddress>,
) {
    for col in &rel.columns {
        collect_addrs_from_linear_relation(col, addrs);
    }
}

fn collect_sorted_unique_addrs(layer: &GKRLayerDescription) -> Vec<GKRAddress> {
    use std::collections::BTreeSet;
    use NoFieldGKRRelation as R;
    let mut addrs = BTreeSet::new();

    for gate in layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
    {
        match &gate.enforced_relation {
            R::EnforceConstraintsMaxQuadratic { input } => {
                for ((a, b), _) in &input.quadratic_terms {
                    addrs.insert(*a);
                    addrs.insert(*b);
                }
                for (addr, _) in &input.linear_terms {
                    addrs.insert(*addr);
                }
            }
            R::Copy { input, .. } => {
                addrs.insert(*input);
            }
            R::InitialGrandProductFromCaches { input, .. }
            | R::TrivialProduct { input, .. } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
            }
            R::UnbalancedGrandProductWithCache {
                scalar, input, ..
            } => {
                addrs.insert(*scalar);
                addrs.insert(*input);
            }
            R::MaskIntoIdentityProduct { input, mask, .. } => {
                addrs.insert(*input);
                addrs.insert(*mask);
            }
            R::MaterializeSingleLookupInput { input, .. } => {
                collect_addrs_from_single_lookup(input, &mut addrs);
            }
            R::MaterializedVectorLookupInput { input, .. } => {
                collect_addrs_from_vector_lookup(input, &mut addrs);
            }
            R::LookupWithCachedDensAndSetup { input, setup, .. } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
                addrs.insert(setup[0]);
                addrs.insert(setup[1]);
            }
            R::LookupPairFromBaseInputs { input, .. } => {
                collect_addrs_from_single_lookup(&input[0], &mut addrs);
                collect_addrs_from_single_lookup(&input[1], &mut addrs);
            }
            R::LookupPairFromMaterializedBaseInputs { input, .. } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
            }
            R::LookupUnbalancedPairWithBaseInputs {
                input, remainder, ..
            } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
                collect_addrs_from_single_lookup(remainder, &mut addrs);
            }
            R::LookupFromBaseInputsWithSetup { input, setup, .. } => {
                collect_addrs_from_single_lookup(input, &mut addrs);
                addrs.insert(setup[0]);
                addrs.insert(setup[1]);
            }
            R::LookupFromMaterializedBaseInputWithSetup { input, setup, .. } => {
                addrs.insert(*input);
                addrs.insert(setup[0]);
                addrs.insert(setup[1]);
            }
            R::LookupUnbalancedPairWithMaterializedBaseInputs {
                input, remainder, ..
            } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
                addrs.insert(*remainder);
            }
            R::LookupPairFromVectorInputs { input, .. } => {
                collect_addrs_from_vector_lookup(&input[0], &mut addrs);
                collect_addrs_from_vector_lookup(&input[1], &mut addrs);
            }
            R::LookupPair { input, .. } => {
                addrs.insert(input[0][0]);
                addrs.insert(input[0][1]);
                addrs.insert(input[1][0]);
                addrs.insert(input[1][1]);
            }
        }
    }
    addrs.into_iter().collect()
}

fn count_unique_addrs(layer: &GKRLayerDescription) -> usize {
    collect_sorted_unique_addrs(layer).len()
}

fn compute_max_pow(layer: &GKRLayerDescription) -> usize {
    use NoFieldGKRRelation as R;
    let mut max_pow = 0usize;
    for gate in layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
    {
        if let R::EnforceConstraintsMaxQuadratic { input } = &gate.enforced_relation {
            for (_, terms) in &input.quadratic_terms {
                for &(_, pow) in terms.iter() {
                    max_pow = max_pow.max(pow);
                }
            }
            for (_, terms) in &input.linear_terms {
                for &(_, pow) in terms.iter() {
                    max_pow = max_pow.max(pow);
                }
            }
            for &(_, pow) in input.constants.iter() {
                max_pow = max_pow.max(pow);
            }
        }
    }
    max_pow
}

fn scan_used_relation_types(layers: &[GKRLayerDescription]) -> (bool, bool, bool) {
    let mut uses_linear = false;
    let mut uses_single_lookup = false;
    let mut uses_vector_lookup = false;

    for layer in layers {
        for gate in layer
            .gates
            .iter()
            .chain(layer.gates_with_external_connections.iter())
        {
            use NoFieldGKRRelation as R;
            match &gate.enforced_relation {
                R::MaterializeSingleLookupInput { .. }
                | R::LookupPairFromBaseInputs { .. }
                | R::LookupUnbalancedPairWithBaseInputs { .. }
                | R::LookupFromBaseInputsWithSetup { .. } => {
                    uses_linear = true;
                    uses_single_lookup = true;
                }
                R::MaterializedVectorLookupInput { .. }
                | R::LookupPairFromVectorInputs { .. } => {
                    uses_linear = true;
                    uses_vector_lookup = true;
                }
                R::Copy { .. }
                | R::InitialGrandProductFromCaches { .. }
                | R::TrivialProduct { .. }
                | R::MaskIntoIdentityProduct { .. }
                | R::LookupFromMaterializedBaseInputWithSetup { .. } 
                | R::LookupPairFromMaterializedBaseInputs { .. } 
                | R::EnforceConstraintsMaxQuadratic { .. }
                | R::UnbalancedGrandProductWithCache { .. }
                | R::LookupWithCachedDensAndSetup { .. }
                | R::LookupUnbalancedPairWithMaterializedBaseInputs { .. }
                | R::LookupPair { .. } => {
                    // No special type imports needed
                }
            }
        }
    }
    (uses_linear, uses_single_lookup, uses_vector_lookup)
}

pub fn generate_gkr_config<F: PrimeField, E: FieldExtension<F> + Field, T>(
    compiled_circuit: &GKRCircuitArtifact<F>,
    proof: &GKRProof<F, E, T>,
    final_trace_size_log_2: usize,
) -> TokenStream
where
    T: ColumnMajorMerkleTreeConstructor<F>,
    [(); E::DEGREE]: Sized,
{
    let num_standard_layers = compiled_circuit.layers.len();
    let initial_layer_for_sumcheck = *proof
        .sumcheck_intermediate_values
        .keys()
        .max()
        .expect("proof must have sumcheck values");

    // Precompute sorted input addresses for each standard layer.
    let standard_sorted_addrs: Vec<Vec<GKRAddress>> = compiled_circuit
        .layers
        .iter()
        .map(|l| collect_sorted_unique_addrs(l))
        .collect();

    // Build the iteration-order addresses for a dim-reducing layer.
    // For the lowest dim-reducing layer, these come directly from global_output_map.
    // For higher layers, they are InnerLayer addresses laid out per output group.
    let build_dim_reducing_addrs = |layer_idx: usize| -> Vec<GKRAddress> {
        let mut addrs = Vec::new();
        if layer_idx == num_standard_layers {
            for (_, group_addrs) in compiled_circuit.global_output_map.iter() {
                for addr in group_addrs.iter() {
                    addrs.push(*addr);
                }
            }
        } else {
            let mut off = 0;
            for (output_type, group_addrs) in compiled_circuit.global_output_map.iter() {
                match output_type {
                    OutputType::PermutationProduct => {
                        for i in 0..group_addrs.len() {
                            addrs.push(GKRAddress::InnerLayer {
                                layer: layer_idx,
                                offset: off + i,
                            });
                        }
                        off += group_addrs.len();
                    }
                    OutputType::Lookup16Bits
                    | OutputType::LookupTimestamps
                    | OutputType::GenericLookup => {
                        addrs.push(GKRAddress::InnerLayer {
                            layer: layer_idx,
                            offset: off,
                        });
                        addrs.push(GKRAddress::InnerLayer {
                            layer: layer_idx,
                            offset: off + 1,
                        });
                        off += 2;
                    }
                }
            }
        }
        addrs
    };

    // Compute sorted addresses for each dim-reducing layer.
    let dim_reducing_sorted_addrs: Vec<Vec<GKRAddress>> = (num_standard_layers
        ..=initial_layer_for_sumcheck)
        .map(|layer_idx| {
            let mut addrs = build_dim_reducing_addrs(layer_idx);
            addrs.sort();
            addrs
        })
        .collect();

    // For standard layer N, output addresses index into the next layer's input space.
    let get_output_sorted_addrs = |layer_idx: usize| -> &[GKRAddress] {
        if layer_idx + 1 < num_standard_layers {
            &standard_sorted_addrs[layer_idx + 1]
        } else if !dim_reducing_sorted_addrs.is_empty() {
            &dim_reducing_sorted_addrs[0]
        } else {
            &standard_sorted_addrs[layer_idx]
        }
    };

    let mut layer_desc_consts = TokenStream::new();
    for (idx, layer_desc) in compiled_circuit.layers.iter().enumerate() {
        layer_desc_consts.extend(transform_layer_description(
            layer_desc,
            idx,
            &standard_sorted_addrs[idx],
            get_output_sorted_addrs(idx),
        ));
    }

    let mut output_groups_stream = TokenStream::new();
    output_groups_stream.append_separated(
        compiled_circuit
            .global_output_map
            .iter()
            .map(|(output_type, addrs)| {
                let ot = transform_output_type(output_type);
                let num = addrs.len();
                quote! {
                    GKROutputGroup { output_type: #ot, num_addresses: #num }
                }
            }),
        quote! {,},
    );

    let mut sorted_addrs_consts = TokenStream::new();
    let mut layer_metas_stream = TokenStream::new();

    // Standard layers
    for layer_idx in 0..num_standard_layers {
        let proof_values = proof
            .sumcheck_intermediate_values
            .get(&layer_idx)
            .expect("missing sumcheck values");
        let num_sumcheck_rounds = proof_values.sumcheck_num_rounds;
        let desc_name = quote::format_ident!("LAYER_{}_DESC", layer_idx);

        let sorted_addrs = &standard_sorted_addrs[layer_idx];
        let sorted_addrs_name = quote::format_ident!("LAYER_{}_SORTED_ADDRS", layer_idx);
        let mut addrs_stream = TokenStream::new();
        addrs_stream.append_separated(
            sorted_addrs.iter().map(|a| transform_gkr_address(a)),
            quote! {,},
        );
        sorted_addrs_consts.extend(quote! {
            const #sorted_addrs_name: &[GKRAddress] = &[#addrs_stream];
        });

        layer_metas_stream.extend(quote! {
            GKRLayerMeta {
                is_dim_reducing: 0usize,
                num_sumcheck_rounds: #num_sumcheck_rounds,
                output_groups: &[],
                layer_desc: Some(&#desc_name),
                sorted_dedup_input_addrs: #sorted_addrs_name,
                input_sorted_indices: &[],
            },
        });
    }

    // Dim-reducing layers
    for (dim_idx, layer_idx) in (num_standard_layers..=initial_layer_for_sumcheck).enumerate() {
        let proof_values = proof
            .sumcheck_intermediate_values
            .get(&layer_idx)
            .expect("missing sumcheck values");
        let num_sumcheck_rounds = proof_values.sumcheck_num_rounds;

        let input_addrs = &dim_reducing_sorted_addrs[dim_idx];

        let sorted_addrs_name = quote::format_ident!("LAYER_{}_SORTED_ADDRS", layer_idx);
        let mut addrs_stream = TokenStream::new();
        addrs_stream.append_separated(
            input_addrs.iter().map(|a| transform_gkr_address(a)),
            quote! {,},
        );
        sorted_addrs_consts.extend(quote! {
            const #sorted_addrs_name: &[GKRAddress] = &[#addrs_stream];
        });

        // input_sorted_indices maps from iteration order to sorted position.
        let iteration_order_addrs = build_dim_reducing_addrs(layer_idx);
        let input_sorted_indices_name =
            quote::format_ident!("LAYER_{}_INPUT_SORTED_IDX", layer_idx);
        let mut idx_stream = TokenStream::new();
        idx_stream.append_separated(
            iteration_order_addrs.iter().map(|addr| {
                let idx = addr_to_idx(addr, input_addrs);
                quote! { #idx }
            }),
            quote! {,},
        );
        sorted_addrs_consts.extend(quote! {
            const #input_sorted_indices_name: &[usize] = &[#idx_stream];
        });

        layer_metas_stream.extend(quote! {
            GKRLayerMeta {
                is_dim_reducing: 1usize,
                num_sumcheck_rounds: #num_sumcheck_rounds,
                output_groups: OUTPUT_GROUPS,
                layer_desc: None,
                sorted_dedup_input_addrs: #sorted_addrs_name,
                input_sorted_indices: #input_sorted_indices_name,
            },
        });
    }

    // Global input addresses (iteration order from global_output_map)
    let mut global_input_addrs_stream = TokenStream::new();
    global_input_addrs_stream.append_separated(
        compiled_circuit
            .global_output_map
            .iter()
            .flat_map(|(_, addrs)| addrs.iter())
            .map(|a| transform_gkr_address(a)),
        quote! {,},
    );

    // Buffer size constants
    let max_sumcheck_rounds = proof
        .sumcheck_intermediate_values
        .values()
        .map(|v| v.sumcheck_num_rounds)
        .max()
        .unwrap_or(0);

    let max_unique_addrs_standard = compiled_circuit
        .layers
        .iter()
        .map(|l| count_unique_addrs(l))
        .max()
        .unwrap_or(0);

    let dim_reducing_addr_count: usize = compiled_circuit
        .global_output_map
        .iter()
        .map(|(_, addrs)| addrs.len())
        .sum();
    let max_addrs = max_unique_addrs_standard.max(dim_reducing_addr_count);

    let max_pow = compiled_circuit
        .layers
        .iter()
        .map(|l| compute_max_pow(l))
        .max()
        .unwrap_or(0)
        + 1;

    let total_output_polys: usize = compiled_circuit
        .global_output_map
        .iter()
        .map(|(_, addrs)| addrs.len())
        .sum();
    let max_evals = total_output_polys * (1usize << final_trace_size_log_2);

    // Aligned buffer for final-step-eval commits: [seed(8) | data | zero-pad to 16-word block]
    // Dim-reducing: 4 evals/addr × DEGREE u32 words; standard: 2 evals/addr × DEGREE
    let degree = E::DEGREE;
    let digest_words = prover::transcript::blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
    let block_words = prover::transcript::blake2s_u32::BLAKE2S_BLOCK_SIZE_U32_WORDS;
    let dim_reducing_words_per_addr = 4 * degree;
    let standard_words_per_addr = 2 * degree;
    let max_data_words = (max_addrs * dim_reducing_words_per_addr)
        .max(max_addrs * standard_words_per_addr)
        .max(max_evals * degree);
    let total = digest_words + max_data_words;
    let eval_buf_size = (total + block_words - 1) / block_words * block_words;

    let (uses_linear, uses_single_lookup, uses_vector_lookup) =
        scan_used_relation_types(&compiled_circuit.layers);

    let mut extra_type_imports = TokenStream::new();
    if uses_linear {
        extra_type_imports.extend(quote! { StaticNoFieldLinearRelation, });
    }
    if uses_single_lookup {
        extra_type_imports.extend(quote! { StaticNoFieldSingleColumnLookupRelation, });
    }
    if uses_vector_lookup {
        extra_type_imports.extend(quote! { StaticNoFieldVectorLookupRelation, });
    }

    let has_inits_teardowns: usize = if proof.inits_and_teardowns_top_bits.is_some() {
        1
    } else {
        0
    };

    let initial_transcript_num_u32_words = {
        let mut tmp = Vec::<u32>::new();
        if let Some(top_bits) = proof.inits_and_teardowns_top_bits {
            tmp.push(top_bits);
        }
        proof.external_challenges.flatten_into_buffer(&mut tmp);
        proof
            .whir_proof
            .setup_commitment
            .commitment
            .cap
            .add_into_buffer(&mut tmp);
        proof
            .whir_proof
            .memory_commitment
            .commitment
            .cap
            .add_into_buffer(&mut tmp);
        proof
            .whir_proof
            .witness_commitment
            .commitment
            .cap
            .add_into_buffer(&mut tmp);
        tmp.len()
    };

    quote! {
        use ::verifier_common::cs::definitions::GKRAddress;
        use ::verifier_common::cs::definitions::gkr_static_types::{
            OutputType, StaticGKRLayerDescription, StaticGateArtifacts, StaticNoFieldGKRRelation,
            StaticNoFieldMaxQuadraticConstraintsGKRRelation,
            #extra_type_imports
        };
        use ::verifier_common::gkr::{GKRVerifierConfig, GKRLayerMeta, GKROutputGroup};

        /// Per-circuit buffer size constants for `verify_gkr_sumcheck`.
        pub const GKR_ROUNDS: usize = #max_sumcheck_rounds;
        pub const GKR_ADDRS: usize = #max_addrs;
        pub const GKR_EVALS: usize = #max_evals;
        pub const GKR_TRANSCRIPT_U32: usize = #initial_transcript_num_u32_words;
        pub const GKR_MAX_POW: usize = #max_pow;
        pub const GKR_EVAL_BUF: usize = #eval_buf_size;

        #layer_desc_consts

        #sorted_addrs_consts

        const OUTPUT_GROUPS: &[GKROutputGroup] = &[#output_groups_stream];

        const LAYER_METAS: &[GKRLayerMeta<'static>] = &[#layer_metas_stream];

        const GLOBAL_INPUT_ADDRS: &[GKRAddress] = &[#global_input_addrs_stream];

        pub const GKR_VERIFIER_CONFIG: GKRVerifierConfig<'static> = GKRVerifierConfig {
            layers: LAYER_METAS,
            has_inits_teardowns: #has_inits_teardowns,
            initial_transcript_num_u32_words: #initial_transcript_num_u32_words,
            final_trace_size_log_2: #final_trace_size_log_2,
            num_standard_layers: #num_standard_layers,
            global_input_addrs: GLOBAL_INPUT_ADDRS,
        };
    }
}
