use proc_macro2::TokenStream;
use quote::quote;

use crate::inlining_generator::MersenneWrapper;
use prover::cs::definitions::GKRAddress;
use prover::cs::gkr_compiler::{GKRLayerDescription, NoFieldGKRRelation};
use prover::field::PrimeField;

use super::addr_to_idx;
use super::constraint_kernel::generate_constraint_kernel;

pub fn generate_layer_compute_claim<MW: MersenneWrapper>(
    layer: &GKRLayerDescription,
    layer_idx: usize,
    output_sorted_addrs: &[GKRAddress],
) -> TokenStream {
    let fn_name = quote::format_ident!("layer_{}_compute_claim", layer_idx);
    let quartic_zero = MW::quartic_zero();
    let quartic_one = MW::quartic_one();

    let mut body = quote! {
        let mut combined = #quartic_zero;
        let mut current_batch = #quartic_one;
    };

    let mul_batch = MW::mul_assign(quote! { current_batch }, quote! { batch_base });

    let gates: Vec<_> = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter())
        .collect();
    let num_gates = gates.len();

    for (gate_idx, gate) in gates.into_iter().enumerate() {
        let is_last = gate_idx == num_gates - 1;
        use NoFieldGKRRelation as R;
        match &gate.enforced_relation {
            R::EnforceConstraintsMaxQuadratic { .. } => {
                if !is_last {
                    body.extend(quote! {
                        #mul_batch;
                    });
                }
            }
            R::Copy { output, .. }
            | R::InitialGrandProductFromCaches { output, .. }
            | R::TrivialProduct { output, .. }
            | R::MaskIntoIdentityProduct { output, .. }
            | R::UnbalancedGrandProductWithCache { output, .. }
            | R::MaterializeSingleLookupInput { output, .. }
            | R::MaterializedVectorLookupInput { output, .. } => {
                let out_idx = addr_to_idx(output, output_sorted_addrs);
                let mul_t = MW::mul_assign(quote! { t }, quote! { claim });
                let add_combined = MW::add_assign(quote! { combined }, quote! { t });
                let advance = if is_last { quote! {} } else { quote! { #mul_batch; } };
                body.extend(quote! {
                    {
                        let bc = current_batch;
                        #advance
                        let claim = output_claims.get(#out_idx);
                        let mut t = bc;
                        #mul_t;
                        #add_combined;
                    }
                });
            }
            R::LookupPair { output, .. }
            | R::LookupPairFromBaseInputs { output, .. }
            | R::LookupPairFromMaterializedBaseInputs { output, .. }
            | R::LookupUnbalancedPairWithBaseInputs { output, .. }
            | R::LookupUnbalancedPairWithMaterializedBaseInputs { output, .. }
            | R::LookupFromBaseInputsWithSetup { output, .. }
            | R::LookupFromMaterializedBaseInputWithSetup { output, .. }
            | R::LookupPairFromVectorInputs { output, .. }
            | R::LookupWithCachedDensAndSetup { output, .. } => {
                let o0 = addr_to_idx(&output[0], output_sorted_addrs);
                let o1 = addr_to_idx(&output[1], output_sorted_addrs);
                let mul_t0 = MW::mul_assign(quote! { t0 }, quote! { c0 });
                let add_t0 = MW::add_assign(quote! { combined }, quote! { t0 });
                let mul_t1 = MW::mul_assign(quote! { t1 }, quote! { c1 });
                let add_t1 = MW::add_assign(quote! { combined }, quote! { t1 });
                let advance = if is_last { quote! {} } else { quote! { #mul_batch; } };
                body.extend(quote! {
                    {
                        let bc0 = current_batch;
                        #mul_batch;
                        let bc1 = current_batch;
                        #advance
                        let c0 = output_claims.get(#o0);
                        let c1 = output_claims.get(#o1);
                        let mut t0 = bc0;
                        #mul_t0;
                        #add_t0;
                        let mut t1 = bc1;
                        #mul_t1;
                        #add_t1;
                    }
                });
            }
        }
    }

    body.extend(quote! { combined });

    let quartic_struct = MW::quartic_struct();
    quote! {
        #[inline(always)]
        unsafe fn #fn_name(
            output_claims: &LazyVec<#quartic_struct, GKR_ADDRS>,
            batch_base: #quartic_struct,
        ) -> #quartic_struct {
            #body
        }
    }
}

pub fn generate_layer_final_step_accumulator<MW: MersenneWrapper, F: PrimeField>(
    layer: &GKRLayerDescription,
    layer_idx: usize,
    input_sorted_addrs: &[GKRAddress],
    max_pow: usize,
) -> TokenStream {
    let fn_name = quote::format_ident!("layer_{}_final_step_accumulator", layer_idx);
    let quartic_zero = MW::quartic_zero();
    let quartic_one = MW::quartic_one();
    let field_one = MW::field_one();

    let quartic_struct = MW::quartic_struct();

    let mut body = quote! {
        let mut acc = [#quartic_zero; 2];
        let mut current_batch = #quartic_one;
    };

    let mul_batch = MW::mul_assign(quote! { current_batch }, quote! { batch_base });

    let gates = layer
        .gates
        .iter()
        .chain(layer.gates_with_external_connections.iter());

    for gate in gates {
        use NoFieldGKRRelation as R;
        match &gate.enforced_relation {
            R::EnforceConstraintsMaxQuadratic { input } => {
                let kernel_body =
                    generate_constraint_kernel::<MW, F>(input, input_sorted_addrs);
                let mul_contrib = MW::mul_assign(quote! { contrib }, quote! { val });
                let add_acc = MW::add_assign(quote! { acc[j] }, quote! { contrib });
                body.extend(quote! {
                    {
                        let bc = current_batch;
                        #mul_batch;
                        for j in 0..2 {
                            let val = { #kernel_body };
                            let mut contrib = bc;
                            #mul_contrib;
                            #add_acc;
                        }
                    }
                });
            }
            R::Copy { input, .. } => {
                let i = addr_to_idx(input, input_sorted_addrs);
                let mul_contrib = MW::mul_assign(quote! { contrib }, quote! { val });
                let add_acc = MW::add_assign(quote! { acc[j] }, quote! { contrib });
                body.extend(quote! {
                    {
                        let bc = current_batch;
                        #mul_batch;
                        for j in 0..2 {
                            let val = unsafe { evals.get_unchecked(#i) }[j];
                            let mut contrib = bc;
                            #mul_contrib;
                            #add_acc;
                        }
                    }
                });
            }
            R::InitialGrandProductFromCaches { input, .. }
            | R::TrivialProduct { input, .. } => {
                let i0 = addr_to_idx(&input[0], input_sorted_addrs);
                let i1 = addr_to_idx(&input[1], input_sorted_addrs);
                let mul_ab = MW::mul_assign(quote! { val }, quote! { vb });
                let mul_contrib = MW::mul_assign(quote! { contrib }, quote! { val });
                let add_acc = MW::add_assign(quote! { acc[j] }, quote! { contrib });
                body.extend(quote! {
                    {
                        let bc = current_batch;
                        #mul_batch;
                        for j in 0..2 {
                            let mut val = unsafe { evals.get_unchecked(#i0) }[j];
                            let vb = unsafe { evals.get_unchecked(#i1) }[j];
                            #mul_ab;
                            let mut contrib = bc;
                            #mul_contrib;
                            #add_acc;
                        }
                    }
                });
            }
            R::MaskIntoIdentityProduct { input, mask, .. } => {
                let i = addr_to_idx(input, input_sorted_addrs);
                let m = addr_to_idx(mask, input_sorted_addrs);
                let sub_one = MW::sub_assign_base(quote! { val }, field_one.clone());
                let mul_mask = MW::mul_assign(quote! { val }, quote! { mask_val });
                let add_one = MW::add_assign_base(quote! { val }, field_one.clone());
                let mul_contrib = MW::mul_assign(quote! { contrib }, quote! { val });
                let add_acc = MW::add_assign(quote! { acc[j] }, quote! { contrib });
                body.extend(quote! {
                    {
                        let bc = current_batch;
                        #mul_batch;
                        for j in 0..2 {
                            let mut val = unsafe { evals.get_unchecked(#i) }[j];
                            let mask_val = unsafe { evals.get_unchecked(#m) }[j];
                            #sub_one;
                            #mul_mask;
                            #add_one;
                            let mut contrib = bc;
                            #mul_contrib;
                            #add_acc;
                        }
                    }
                });
            }
            R::LookupPair { input, .. } => {
                let i00 = addr_to_idx(&input[0][0], input_sorted_addrs);
                let i01 = addr_to_idx(&input[0][1], input_sorted_addrs);
                let i10 = addr_to_idx(&input[1][0], input_sorted_addrs);
                let i11 = addr_to_idx(&input[1][1], input_sorted_addrs);
                generate_two_output_body::<MW>(
                    &mut body, &mul_batch,
                    quote! {
                        let a = unsafe { evals.get_unchecked(#i00) }[j];
                        let b = unsafe { evals.get_unchecked(#i01) }[j];
                        let c = unsafe { evals.get_unchecked(#i10) }[j];
                        let d = unsafe { evals.get_unchecked(#i11) }[j];
                    },
                    // num = a*d + c*b
                    |mw_mul, mw_add| {
                        let mul_ad = mw_mul(quote! { num }, quote! { d });
                        let mul_cb = mw_mul(quote! { cb_tmp }, quote! { b });
                        let add_cb = mw_add(quote! { num }, quote! { cb_tmp });
                        quote! {
                            let mut num = a;
                            #mul_ad;
                            let mut cb_tmp = c;
                            #mul_cb;
                            #add_cb;
                            num
                        }
                    },
                    // den = b*d
                    |mw_mul, _| {
                        let mul_bd = mw_mul(quote! { den }, quote! { d });
                        quote! {
                            let mut den = b;
                            #mul_bd;
                            den
                        }
                    },
                );
            }
            R::LookupPairFromMaterializedBaseInputs { input, .. } => {
                let i0 = addr_to_idx(&input[0], input_sorted_addrs);
                let i1 = addr_to_idx(&input[1], input_sorted_addrs);
                generate_two_output_body::<MW>(
                    &mut body, &mul_batch,
                    quote! {
                        let mut b_g = unsafe { evals.get_unchecked(#i0) }[j];
                        let mut d_g = unsafe { evals.get_unchecked(#i1) }[j];
                    },
                    |_, mw_add| {
                        let add_gamma_b = mw_add(quote! { b_g }, quote! { lookup_additive_challenge });
                        let add_gamma_d = mw_add(quote! { d_g }, quote! { lookup_additive_challenge });
                        let add_bd = mw_add(quote! { num }, quote! { d_g });
                        quote! {
                            #add_gamma_b;
                            #add_gamma_d;
                            let mut num = b_g;
                            #add_bd;
                            num
                        }
                    },
                    |mw_mul, _| {
                        let mul_bd = mw_mul(quote! { den }, quote! { d_g });
                        quote! {
                            let mut den = b_g;
                            #mul_bd;
                            den
                        }
                    },
                );
            }
            R::LookupFromMaterializedBaseInputWithSetup { input, setup, .. } => {
                let i_in = addr_to_idx(input, input_sorted_addrs);
                let s0 = addr_to_idx(&setup[0], input_sorted_addrs);
                let s1 = addr_to_idx(&setup[1], input_sorted_addrs);
                generate_two_output_body::<MW>(
                    &mut body, &mul_batch,
                    quote! {
                        let mut b_g = unsafe { evals.get_unchecked(#i_in) }[j];
                        let mut d_g = unsafe { evals.get_unchecked(#s1) }[j];
                        let mut cb_g = unsafe { evals.get_unchecked(#s0) }[j];
                    },
                    |mw_mul, mw_add| {
                        let add_gamma_b = mw_add(quote! { b_g }, quote! { lookup_additive_challenge });
                        let add_gamma_d = mw_add(quote! { d_g }, quote! { lookup_additive_challenge });
                        let mul_cb = mw_mul(quote! { cb_g }, quote! { b_g });
                        let sub_cb = MW::sub_assign(quote! { num }, quote! { cb_g });
                        quote! {
                            #add_gamma_b;
                            #add_gamma_d;
                            #mul_cb;
                            let mut num = d_g;
                            #sub_cb;
                            num
                        }
                    },
                    |mw_mul, _| {
                        let mul_bd = mw_mul(quote! { den }, quote! { d_g });
                        quote! {
                            let mut den = b_g;
                            #mul_bd;
                            den
                        }
                    },
                );
            }
            R::LookupUnbalancedPairWithMaterializedBaseInputs {
                input, remainder, ..
            } => {
                let i0 = addr_to_idx(&input[0], input_sorted_addrs);
                let i1 = addr_to_idx(&input[1], input_sorted_addrs);
                let r = addr_to_idx(remainder, input_sorted_addrs);
                generate_two_output_body::<MW>(
                    &mut body, &mul_batch,
                    quote! {
                        let a = unsafe { evals.get_unchecked(#i0) }[j];
                        let b = unsafe { evals.get_unchecked(#i1) }[j];
                        let mut d_g = unsafe { evals.get_unchecked(#r) }[j];
                    },
                    |mw_mul, mw_add| {
                        let add_gamma = mw_add(quote! { d_g }, quote! { lookup_additive_challenge });
                        let mul_ad = mw_mul(quote! { num }, quote! { d_g });
                        let add_b = mw_add(quote! { num }, quote! { b });
                        quote! {
                            #add_gamma;
                            let mut num = a;
                            #mul_ad;
                            #add_b;
                            num
                        }
                    },
                    |mw_mul, _| {
                        let mul_bd = mw_mul(quote! { den }, quote! { d_g });
                        quote! {
                            let mut den = b;
                            #mul_bd;
                            den
                        }
                    },
                );
            }
            R::LookupWithCachedDensAndSetup { input, setup, .. } => {
                let i0 = addr_to_idx(&input[0], input_sorted_addrs);
                let i1 = addr_to_idx(&input[1], input_sorted_addrs);
                let s0 = addr_to_idx(&setup[0], input_sorted_addrs);
                let s1 = addr_to_idx(&setup[1], input_sorted_addrs);
                generate_two_output_body::<MW>(
                    &mut body, &mul_batch,
                    quote! {
                        let a = unsafe { evals.get_unchecked(#i0) }[j];
                        let b = unsafe { evals.get_unchecked(#i1) }[j];
                        let c = unsafe { evals.get_unchecked(#s0) }[j];
                        let d = unsafe { evals.get_unchecked(#s1) }[j];
                    },
                    |mw_mul, _| {
                        let mul_ad = mw_mul(quote! { ad }, quote! { d });
                        let mul_cb = mw_mul(quote! { cb }, quote! { b });
                        let sub_cb = MW::sub_assign(quote! { ad }, quote! { cb });
                        quote! {
                            let mut ad = a;
                            #mul_ad;
                            let mut cb = c;
                            #mul_cb;
                            #sub_cb;
                            ad
                        }
                    },
                    |mw_mul, _| {
                        let mul_bd = mw_mul(quote! { den }, quote! { d });
                        quote! {
                            let mut den = b;
                            #mul_bd;
                            den
                        }
                    },
                );
            }
            _ => {
                panic!(
                    "Unimplemented relation variant in GKR inlining generator: {:?}",
                    gate.enforced_relation
                );
            }
        }
    }

    body.extend(quote! { acc });

    let field_struct = MW::field_struct();
    let quartic_struct = MW::quartic_struct();
    quote! {
        #[inline(always)]
        unsafe fn #fn_name(
            evals: &[[#quartic_struct; 2]],
            batch_base: #quartic_struct,
            lookup_additive_challenge: #quartic_struct,
            challenge_powers: &[#quartic_struct; GKR_MAX_POW],
        ) -> [#quartic_struct; 2] {
            #body
        }
    }
}

fn generate_two_output_body<MW: MersenneWrapper>(
    body: &mut TokenStream,
    mul_batch: &TokenStream,
    setup_vars: TokenStream,
    gen_num: impl FnOnce(
        fn(TokenStream, TokenStream) -> TokenStream,
        fn(TokenStream, TokenStream) -> TokenStream,
    ) -> TokenStream,
    gen_den: impl FnOnce(
        fn(TokenStream, TokenStream) -> TokenStream,
        fn(TokenStream, TokenStream) -> TokenStream,
    ) -> TokenStream,
) {
    let num_expr = gen_num(MW::mul_assign, MW::add_assign);
    let den_expr = gen_den(MW::mul_assign, MW::add_assign);
    let mul_c0 = MW::mul_assign(quote! { c0 }, quote! { out0 });
    let mul_c1 = MW::mul_assign(quote! { c1 }, quote! { out1 });
    let add_c0 = MW::add_assign(quote! { acc[j] }, quote! { c0 });
    let add_c1 = MW::add_assign(quote! { acc[j] }, quote! { c1 });
    body.extend(quote! {
        {
            let bc0 = current_batch;
            #mul_batch;
            let bc1 = current_batch;
            #mul_batch;
            for j in 0..2 {
                #setup_vars
                let out0 = { #num_expr };
                let out1 = { #den_expr };
                let mut c0 = bc0;
                #mul_c0;
                let mut c1 = bc1;
                #mul_c1;
                #add_c0;
                #add_c1;
            }
        }
    });
}
