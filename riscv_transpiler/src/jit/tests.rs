use super::*;
use crate::vm::test::*;
use std::path::Path;

#[test]
fn test_jit_simple_fibonacci() {
    let path = std::env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    // let (_, binary) = read_binary(&Path::new("riscv_transpiler/examples/fibonacci/app.bin"));
    // let (_, text) = read_binary(&Path::new("riscv_transpiler/examples/fibonacci/app.text"));

    // let (_, binary) = read_binary(&Path::new("examples/fibonacci/app.bin"));
    // let (_, text) = read_binary(&Path::new("examples/fibonacci/app.text"));

    let (_, binary) = read_binary(&Path::new("examples/keccak_f1600/app.bin"));
    let (_, text) = read_binary(&Path::new("examples/keccak_f1600/app.text"));

    run_alternative_simulator(&text, &mut (), &binary);
}
