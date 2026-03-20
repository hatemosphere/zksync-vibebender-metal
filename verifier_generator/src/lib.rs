#![expect(warnings)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use ::prover::*;
use prover::field::*;

mod utils;
use self::utils::*;

pub mod generator;
pub use self::generator::*;

pub mod inlining_generator;
pub use self::inlining_generator::*;

pub mod mersenne_wrapper;
pub use self::mersenne_wrapper::*;

pub mod gkr_inlining;

pub fn generate_from_reader<R: std::io::Read>(reader: R) -> (String, String) {
    let description = serde_json::from_reader(reader).unwrap();
    generate_for_description(description)
}

pub fn generate_for_description(
    description: CompiledCircuitArtifact<Mersenne31Field>,
) -> (String, String) {
    let layout = generate_from_parts(&description);

    let quotient = generate_inlined(description);

    (layout.to_string(), quotient.to_string())
}

// mod testing_file;
// mod testing_inlining_file;

#[cfg(test)]
mod test {
    use test_utils::skip_if_ci;

    use std::io::Write;

    use super::*;

    fn deserialize_from_file<T: serde::de::DeserializeOwned>(filename: &str) -> T {
        let src = std::fs::File::open(filename).unwrap();
        serde_json::from_reader(src).unwrap()
    }

    #[cfg(test)]
    #[test]
    #[serial_test::serial]
    fn launch() {
        skip_if_ci!();
        let compiled_circuit = deserialize_from_file("../prover/full_machine_layout.json");
        // let compiled_circuit = deserialize_from_file("../prover/blake2s_delegation_circuit_layout.json");
        // let compiled_circuit =
        //     deserialize_from_file("../prover/keccak_delegation_circuit_layout.json");

        let result = generate_from_parts(&compiled_circuit);

        let mut dst = std::fs::File::create("./src/generated.rs").unwrap();
        dst.write_all(&result.to_string().as_bytes()).unwrap();
    }

    #[cfg(test)]
    #[test]
    #[serial_test::serial]
    fn launch_inlining() {
        skip_if_ci!();
        let compiled_circuit = deserialize_from_file("../prover/full_machine_layout.json");
        // let compiled_circuit =
        //     deserialize_from_file("../prover/blake2s_delegation_circuit_layout.json");
        // let compiled_circuit =
        //     deserialize_from_file("../prover/keccak_delegation_circuit_layout.json");

        let result = generate_inlined(compiled_circuit);

        let mut dst = std::fs::File::create("./src/generated_inlined_verifier.rs").unwrap();
        dst.write_all(&result.to_string().as_bytes()).unwrap();
    }

    #[cfg(test)]
    #[test]
    #[serial_test::serial]
    fn generate_for_unrolled_circuits() {
        skip_if_ci!();
        let circuit_names = vec![
            "add_sub_lui_auipc_mop_preprocessed",
            "jump_branch_slt_preprocessed",
            "shift_binop_csrrw_preprocessed",
            "mul_div_preprocessed",
            "mul_div_unsigned_preprocessed",
            "load_store_preprocessed",
            "word_only_load_store_preprocessed",
            "subword_only_load_store_preprocessed",
            "inits_and_teardowns_preprocessed",
        ];

        for name in circuit_names {
            let compiled_circuit = deserialize_from_file(&format!("../cs/{}_layout.json", name));

            let result = generate_from_parts(&compiled_circuit);
            let mut dst = std::fs::File::create(format!("./generated/{}_layout.rs", name)).unwrap();
            dst.write_all(&result.to_string().as_bytes()).unwrap();

            let result = generate_inlined(compiled_circuit);
            let mut dst =
                std::fs::File::create(format!("./generated/{}_quotient.rs", name)).unwrap();
            dst.write_all(&result.to_string().as_bytes()).unwrap();
        }
    }

    #[test]
    fn generate_gkr_inlined() {
        use crate::inlining_generator::DefaultBabyBearField;
        use prover::cs::gkr_compiler::GKRCircuitArtifact;
        use prover::field::baby_bear::base::BabyBearField;
        use prover::field::baby_bear::ext4::BabyBearExt4;
        use prover::gkr::prover::GKRProof;
        use prover::merkle_trees::DefaultTreeConstructor;

        let circuit_names = vec!["add_sub_lui_auipc_mop"];

        for name in circuit_names {
            let compiled_circuit: GKRCircuitArtifact<BabyBearField> =
                deserialize_from_file(&format!("../prover/{}_gkr_circuit.json", name));
            let proof: GKRProof<BabyBearField, BabyBearExt4, DefaultTreeConstructor> =
                deserialize_from_file(&format!("../prover/{}_gkr_proof.json", name));

            let result = gkr_inlining::generate_gkr_inlined::<DefaultBabyBearField, _, _, _>(
                &compiled_circuit,
                &proof,
                4,
            );

            let path = format!("../verifier/src/generated/gkr_verifier.rs");
            let mut dst = std::fs::File::create(&path).unwrap();
            dst.write_all(&result.to_string().as_bytes()).unwrap();
            drop(dst);
            std::process::Command::new("rustfmt")
                .arg(&path)
                .status()
                .ok();
        }
    }

    #[cfg(feature = "legacy_tests")]
    #[test]
    #[serial_test::serial]
    // TODO(legacy-cleanup): determine whether the legacy code path exercised here can be removed.
    fn generate_reduced_machine() {
        skip_if_ci!();
        let compiled_circuit = deserialize_from_file("../prover/reduced_machine_layout");

        let result = generate_from_parts(&compiled_circuit);
        let mut dst = std::fs::File::create("./src/generated.rs").unwrap();
        dst.write_all(&result.to_string().as_bytes()).unwrap();

        let result = generate_inlined(compiled_circuit);
        let mut dst = std::fs::File::create("./src/generated_inlined_verifier.rs").unwrap();
        dst.write_all(&result.to_string().as_bytes()).unwrap();
    }
}
