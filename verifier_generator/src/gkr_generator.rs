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
use prover::field::{Field, FieldExtension, PrimeField};
use prover::gkr::prover::{GKRExternalChallenges, GKRProof};
use prover::merkle_trees::ColumnMajorMerkleTreeConstructor;

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

fn transform_linear_relation(rel: &NoFieldLinearRelation) -> TokenStream {
    let constant = rel.constant;
    let mut terms_stream = TokenStream::new();
    terms_stream.append_separated(
        rel.linear_terms.iter().map(|(coeff, addr)| {
            let addr = transform_gkr_address(addr);
            quote! { (#coeff, #addr) }
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

fn transform_single_column_lookup(rel: &NoFieldSingleColumnLookupRelation) -> TokenStream {
    let input = transform_linear_relation(&rel.input);
    let lookup_set_index = rel.lookup_set_index;
    quote! {
        StaticNoFieldSingleColumnLookupRelation {
            input: #input,
            lookup_set_index: #lookup_set_index,
        }
    }
}

fn transform_vector_lookup(rel: &NoFieldVectorLookupRelation) -> TokenStream {
    let mut columns_stream = TokenStream::new();
    columns_stream.append_separated(
        rel.columns.iter().map(|c| transform_linear_relation(c)),
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
) -> TokenStream {
    let mut qt_stream = TokenStream::new();
    qt_stream.append_separated(
        rel.quadratic_terms.iter().map(|((a, b), terms)| {
            let a = transform_gkr_address(a);
            let b = transform_gkr_address(b);
            let mut inner = TokenStream::new();
            inner.append_separated(
                terms.iter().map(|(coeff, pow)| {
                    quote! { (#coeff, #pow) }
                }),
                quote! {,},
            );
            quote! { ((#a, #b), &[#inner] as &[(u32, usize)]) }
        }),
        quote! {,},
    );

    let mut lt_stream = TokenStream::new();
    lt_stream.append_separated(
        rel.linear_terms.iter().map(|(addr, terms)| {
            let addr = transform_gkr_address(addr);
            let mut inner = TokenStream::new();
            inner.append_separated(
                terms.iter().map(|(coeff, pow)| {
                    quote! { (#coeff, #pow) }
                }),
                quote! {,},
            );
            quote! { (#addr, &[#inner] as &[(u32, usize)]) }
        }),
        quote! {,},
    );

    let mut ct_stream = TokenStream::new();
    ct_stream.append_separated(
        rel.constants.iter().map(|(coeff, pow)| {
            quote! { (#coeff, #pow) }
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

fn transform_relation(rel: &NoFieldGKRRelation) -> TokenStream {
    use NoFieldGKRRelation as R;
    match rel {
        R::EnforceConstraintsMaxQuadratic { input } => {
            let input = transform_max_quadratic_relation(input);
            quote! {
                StaticNoFieldGKRRelation::EnforceConstraintsMaxQuadratic { input: #input }
            }
        }
        R::Copy { input, output } => {
            let i = transform_gkr_address(input);
            let o = transform_gkr_address(output);
            quote! { StaticNoFieldGKRRelation::Copy { input: #i, output: #o } }
        }
        R::InitialGrandProductFromCaches { input, output } => {
            let i0 = transform_gkr_address(&input[0]);
            let i1 = transform_gkr_address(&input[1]);
            let o = transform_gkr_address(output);
            quote! { StaticNoFieldGKRRelation::InitialGrandProductFromCaches { input: [#i0, #i1], output: #o } }
        }
        R::UnbalancedGrandProductWithCache {
            scalar,
            input,
            output,
        } => {
            let s = transform_gkr_address(scalar);
            let i = transform_gkr_address(input);
            let o = transform_gkr_address(output);
            quote! { StaticNoFieldGKRRelation::UnbalancedGrandProductWithCache { scalar: #s, input: #i, output: #o } }
        }
        R::TrivialProduct { input, output } => {
            let i0 = transform_gkr_address(&input[0]);
            let i1 = transform_gkr_address(&input[1]);
            let o = transform_gkr_address(output);
            quote! { StaticNoFieldGKRRelation::TrivialProduct { input: [#i0, #i1], output: #o } }
        }
        R::MaskIntoIdentityProduct {
            input,
            mask,
            output,
        } => {
            let i = transform_gkr_address(input);
            let m = transform_gkr_address(mask);
            let o = transform_gkr_address(output);
            quote! { StaticNoFieldGKRRelation::MaskIntoIdentityProduct { input: #i, mask: #m, output: #o } }
        }
        R::MaterializeSingleLookupInput { input, output } => {
            let i = transform_single_column_lookup(input);
            let o = transform_gkr_address(output);
            quote! { StaticNoFieldGKRRelation::MaterializeSingleLookupInput { input: #i, output: #o } }
        }
        R::MaterializedVectorLookupInput { input, output } => {
            let i = transform_vector_lookup(input);
            let o = transform_gkr_address(output);
            quote! { StaticNoFieldGKRRelation::MaterializedVectorLookupInput { input: #i, output: #o } }
        }
        R::LookupWithCachedDensAndSetup {
            input,
            setup,
            output,
        } => {
            let i0 = transform_gkr_address(&input[0]);
            let i1 = transform_gkr_address(&input[1]);
            let s0 = transform_gkr_address(&setup[0]);
            let s1 = transform_gkr_address(&setup[1]);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupWithCachedDensAndSetup { input: [#i0, #i1], setup: [#s0, #s1], output: [#o0, #o1] } }
        }
        R::LookupPairFromBaseInputs { input, output } => {
            let i0 = transform_single_column_lookup(&input[0]);
            let i1 = transform_single_column_lookup(&input[1]);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPairFromBaseInputs { input: [#i0, #i1], output: [#o0, #o1] } }
        }
        R::LookupPairFromMaterializedBaseInputs { input, output } => {
            let i0 = transform_gkr_address(&input[0]);
            let i1 = transform_gkr_address(&input[1]);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPairFromMaterializedBaseInputs { input: [#i0, #i1], output: [#o0, #o1] } }
        }
        R::LookupUnbalancedPairWithBaseInputs {
            input,
            remainder,
            output,
        } => {
            let i0 = transform_gkr_address(&input[0]);
            let i1 = transform_gkr_address(&input[1]);
            let r = transform_single_column_lookup(remainder);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupUnbalancedPairWithBaseInputs { input: [#i0, #i1], remainder: #r, output: [#o0, #o1] } }
        }
        R::LookupFromBaseInputsWithSetup {
            input,
            setup,
            output,
        } => {
            let i = transform_single_column_lookup(input);
            let s0 = transform_gkr_address(&setup[0]);
            let s1 = transform_gkr_address(&setup[1]);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupFromBaseInputsWithSetup { input: #i, setup: [#s0, #s1], output: [#o0, #o1] } }
        }
        R::LookupFromMaterializedBaseInputWithSetup {
            input,
            setup,
            output,
        } => {
            let i = transform_gkr_address(input);
            let s0 = transform_gkr_address(&setup[0]);
            let s1 = transform_gkr_address(&setup[1]);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupFromMaterializedBaseInputWithSetup { input: #i, setup: [#s0, #s1], output: [#o0, #o1] } }
        }
        R::LookupUnbalancedPairWithMaterializedBaseInputs {
            input,
            remainder,
            output,
        } => {
            let i0 = transform_gkr_address(&input[0]);
            let i1 = transform_gkr_address(&input[1]);
            let r = transform_gkr_address(remainder);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupUnbalancedPairWithMaterializedBaseInputs { input: [#i0, #i1], remainder: #r, output: [#o0, #o1] } }
        }
        R::LookupPairFromVectorInputs { input, output } => {
            let i0 = transform_vector_lookup(&input[0]);
            let i1 = transform_vector_lookup(&input[1]);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPairFromVectorInputs { input: [#i0, #i1], output: [#o0, #o1] } }
        }
        R::LookupPair { input, output } => {
            let i00 = transform_gkr_address(&input[0][0]);
            let i01 = transform_gkr_address(&input[0][1]);
            let i10 = transform_gkr_address(&input[1][0]);
            let i11 = transform_gkr_address(&input[1][1]);
            let o0 = transform_gkr_address(&output[0]);
            let o1 = transform_gkr_address(&output[1]);
            quote! { StaticNoFieldGKRRelation::LookupPair { input: [[#i00, #i01], [#i10, #i11]], output: [#o0, #o1] } }
        }
    }
}

fn transform_gate(gate: &GateArtifacts) -> TokenStream {
    let output_layer = gate.output_layer;
    let rel = transform_relation(&gate.enforced_relation);
    quote! {
        StaticGateArtifacts {
            output_layer: #output_layer,
            enforced_relation: #rel,
        }
    }
}

fn transform_layer_description(layer: &GKRLayerDescription, layer_idx: usize) -> TokenStream {
    let gates_name = quote::format_ident!("LAYER_{}_GATES", layer_idx);
    let ext_gates_name = quote::format_ident!("LAYER_{}_EXT_GATES", layer_idx);
    let base_openings_name = quote::format_ident!("LAYER_{}_BASE_OPENINGS", layer_idx);
    let desc_name = quote::format_ident!("LAYER_{}_DESC", layer_idx);

    let mut gates_stream = TokenStream::new();
    gates_stream.append_separated(layer.gates.iter().map(|g| transform_gate(g)), quote! {,});

    let mut ext_gates_stream = TokenStream::new();
    ext_gates_stream.append_separated(
        layer
            .gates_with_external_connections
            .iter()
            .map(|g| transform_gate(g)),
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

fn compute_max_pow(layer: &GKRLayerDescription) -> usize {
    use NoFieldGKRRelation as R;
    let mut max_pow = 0usize;
    let relations = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
        .map(|g| &g.enforced_relation);
    for relation in relations {
        if let R::EnforceConstraintsMaxQuadratic { input } = relation {
            for (_, terms) in input.quadratic_terms.iter() {
                for &(_, pow) in terms.iter() {
                    max_pow = max_pow.max(pow);
                }
            }
            for (_, terms) in input.linear_terms.iter() {
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

fn count_raw_addrs(layer: &GKRLayerDescription) -> usize {
    use NoFieldGKRRelation as R;
    let mut count = 0;
    let relations = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
        .map(|g| &g.enforced_relation);
    for relation in relations {
        count += match relation {
            R::EnforceConstraintsMaxQuadratic { input } => {
                input.linear_terms.len() + input.quadratic_terms.len() * 2
            }
            R::Copy { .. } => 1,
            R::InitialGrandProductFromCaches { .. } | R::TrivialProduct { .. } => 2,
            R::MaskIntoIdentityProduct { .. } => 2,
            R::LookupPair { .. } => 4,
            R::LookupPairFromMaterializedBaseInputs { .. } => 2,
            R::LookupFromMaterializedBaseInputWithSetup { .. } => 3,
            R::LookupUnbalancedPairWithMaterializedBaseInputs { .. } => 3,
            R::LookupWithCachedDensAndSetup { .. } => 4,
            _ => panic!("unimplemented relation variant in count_raw_addrs"),
        };
    }
    count
}

fn count_unique_addrs(layer: &GKRLayerDescription) -> usize {
    use std::collections::BTreeSet;
    use NoFieldGKRRelation as R;
    let mut addrs = BTreeSet::new();
    let relations = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
        .map(|g| &g.enforced_relation);
    for relation in relations {
        match relation {
            R::EnforceConstraintsMaxQuadratic { input } => {
                for (addr, _) in input.linear_terms.iter() {
                    addrs.insert(*addr);
                }
                for ((a, b), _) in input.quadratic_terms.iter() {
                    addrs.insert(*a);
                    addrs.insert(*b);
                }
            }
            R::Copy { input, .. } => {
                addrs.insert(*input);
            }
            R::InitialGrandProductFromCaches { input, .. } | R::TrivialProduct { input, .. } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
            }
            R::MaskIntoIdentityProduct { input, mask, .. } => {
                addrs.insert(*input);
                addrs.insert(*mask);
            }
            R::LookupPair { input, .. } => {
                addrs.insert(input[0][0]);
                addrs.insert(input[0][1]);
                addrs.insert(input[1][0]);
                addrs.insert(input[1][1]);
            }
            R::LookupPairFromMaterializedBaseInputs { input, .. } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
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
            R::LookupWithCachedDensAndSetup { input, setup, .. } => {
                addrs.insert(input[0]);
                addrs.insert(input[1]);
                addrs.insert(setup[0]);
                addrs.insert(setup[1]);
            }
            _ => panic!("unimplemented relation variant in count_unique_addrs"),
        }
    }
    addrs.len()
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

    let mut layer_desc_consts = TokenStream::new();
    for (idx, layer_desc) in compiled_circuit.layers.iter().enumerate() {
        layer_desc_consts.extend(transform_layer_description(layer_desc, idx));
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

    let mut layer_metas_stream = TokenStream::new();

    // Standard layers
    for layer_idx in 0..num_standard_layers {
        let proof_values = proof
            .sumcheck_intermediate_values
            .get(&layer_idx)
            .expect("missing sumcheck values");
        let num_sumcheck_rounds = proof_values.sumcheck_num_rounds;
        let desc_name = quote::format_ident!("LAYER_{}_DESC", layer_idx);
        layer_metas_stream.extend(quote! {
            GKRLayerMeta {
                is_dim_reducing: 0usize,
                num_sumcheck_rounds: #num_sumcheck_rounds,
                output_groups: &[],
                layer_desc: Some(&#desc_name),
            },
        });
    }

    // Dim-reducing layers
    for layer_idx in num_standard_layers..=initial_layer_for_sumcheck {
        let proof_values = proof
            .sumcheck_intermediate_values
            .get(&layer_idx)
            .expect("missing sumcheck values");
        let num_sumcheck_rounds = proof_values.sumcheck_num_rounds;
        layer_metas_stream.extend(quote! {
            GKRLayerMeta {
                is_dim_reducing: 1usize,
                num_sumcheck_rounds: #num_sumcheck_rounds,
                output_groups: OUTPUT_GROUPS,
                layer_desc: None,
            },
        });
    }

    // Build global_input_addrs
    let mut global_input_addrs_stream = TokenStream::new();
    global_input_addrs_stream.append_separated(
        compiled_circuit
            .global_output_map
            .iter()
            .flat_map(|(_, addrs)| addrs.iter())
            .map(|a| transform_gkr_address(a)),
        quote! {,},
    );

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

    // Dim-reducing layers use global_output_map addresses
    let dim_reducing_addr_count: usize = compiled_circuit
        .global_output_map
        .iter()
        .map(|(_, addrs)| addrs.len())
        .sum();
    let max_addrs = max_unique_addrs_standard.max(dim_reducing_addr_count);

    // Max raw (pre-dedup) addresses across standard layers
    let max_raw_addrs = compiled_circuit
        .layers
        .iter()
        .map(|l| count_raw_addrs(l))
        .max()
        .unwrap_or(0);

    // Max challenge power index across all constraint relations (+1 for array size)
    let max_pow = compiled_circuit
        .layers
        .iter()
        .map(|l| compute_max_pow(l))
        .max()
        .unwrap_or(0)
        + 1;

    // Max extension field elements needed for eval buffers
    let total_output_polys: usize = compiled_circuit
        .global_output_map
        .iter()
        .map(|(_, addrs)| addrs.len())
        .sum();
    let evals_per_poly = 1usize << final_trace_size_log_2;
    let max_evals = total_output_polys * evals_per_poly;

    // Scan all relations to determine which static types are actually used
    let mut uses_linear_relation = false;
    let mut uses_single_column_lookup = false;
    let mut uses_vector_lookup = false;
    for layer in compiled_circuit.layers.iter() {
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
                    uses_linear_relation = true;
                    uses_single_column_lookup = true;
                }
                R::MaterializedVectorLookupInput { .. } | R::LookupPairFromVectorInputs { .. } => {
                    uses_linear_relation = true;
                    uses_vector_lookup = true;
                }
                R::EnforceConstraintsMaxQuadratic { .. } => {
                    // uses StaticNoFieldMaxQuadraticConstraintsGKRRelation (always imported)
                }
                _ => {}
            }
        }
    }

    let mut extra_type_imports = TokenStream::new();
    if uses_linear_relation {
        extra_type_imports.extend(quote! { StaticNoFieldLinearRelation, });
    }
    if uses_single_column_lookup {
        extra_type_imports.extend(quote! { StaticNoFieldSingleColumnLookupRelation, });
    }
    if uses_vector_lookup {
        extra_type_imports.extend(quote! { StaticNoFieldVectorLookupRelation, });
    }

    let has_inits_teardowns: usize = if proof.inits_and_teardowns_top_bits.is_some() { 1 } else { 0 };

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
        pub const GKR_RAW_ADDRS: usize = #max_raw_addrs;
        pub const GKR_EVALS: usize = #max_evals;
        pub const GKR_TRANSCRIPT_U32: usize = #initial_transcript_num_u32_words;
        pub const GKR_MAX_POW: usize = #max_pow;

        #layer_desc_consts

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
