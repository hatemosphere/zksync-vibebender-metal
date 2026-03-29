#![allow(unexpected_cfgs)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_metal)");

    if cfg!(not(target_os = "macos")) {
        println!("cargo::warning=Metal GPU prover is only supported on macOS");
        println!("cargo::rustc-cfg=no_metal");
        return;
    }

    // Check if xcrun metal compiler is available
    let metal_check = Command::new("xcrun")
        .args(["--find", "metal"])
        .output();

    match metal_check {
        Ok(output) if output.status.success() => {}
        _ => {
            println!("cargo::warning=Metal compiler (xcrun metal) not found. Install Xcode Command Line Tools.");
            println!("cargo::rustc-cfg=no_metal");
            return;
        }
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = Path::new(&manifest_dir).join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Collect all .metal files
    let all_metal_files: Vec<PathBuf> = glob::glob(shader_dir.join("**/*.metal").to_str().unwrap())
        .expect("Failed to read shader glob pattern")
        .filter_map(|entry| entry.ok())
        .collect();

    // Find files that are #include'd by other files (these are headers, not standalone units)
    let mut included_files = std::collections::HashSet::new();
    for file in &all_metal_files {
        if let Ok(contents) = std::fs::read_to_string(file) {
            for line in contents.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("#include") {
                    let rest = rest.trim();
                    if let Some(path) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                        if let Some(parent) = file.parent() {
                            if let Ok(resolved) = parent.join(path).canonicalize() {
                                included_files.insert(resolved);
                            }
                        }
                    }
                }
            }
        }
    }

    // Only compile files that are not included by other files (standalone compilation units)
    let metal_files: Vec<PathBuf> = all_metal_files
        .into_iter()
        .filter(|f| {
            if let Ok(canonical) = f.canonicalize() {
                !included_files.contains(&canonical)
            } else {
                true
            }
        })
        .collect();

    if metal_files.is_empty() {
        println!("cargo::warning=No .metal shader files found in shaders/");
        // Create an empty metallib so compilation succeeds
        std::fs::write(out_dir.join("gpu_prover_metal.metallib"), &[]).unwrap();
        return;
    }

    // Rerun if any shader changes
    for file in &metal_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    // Compile each .metal file to .air
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        // Include parent dir name to avoid collisions between same-named files in subdirs
        let parent = metal_file
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("root");
        let air_name = if parent == "shaders" {
            format!("{stem}.air")
        } else {
            format!("{parent}_{stem}.air")
        };
        let air_file = out_dir.join(&air_name);

        // blake2s rounds are now manually unrolled (no SIGMAS loop), so the
        // Metal compiler optimization bug that required -O0 should no longer apply.
        let opt_level = "-O2";
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                "-std=metal3.0",
                opt_level,
                // Include the shaders directory for cross-file #include
                "-I",
                shader_dir.to_str().unwrap(),
                metal_file.to_str().unwrap(),
                "-o",
                air_file.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run xcrun metal compiler");

        if !status.success() {
            panic!(
                "Metal shader compilation failed for {}",
                metal_file.display()
            );
        }
        air_files.push(air_file);
    }

    // Link all .air files into a single .metallib
    let metallib_path = out_dir.join("gpu_prover_metal.metallib");
    let mut link_cmd = Command::new("xcrun");
    link_cmd
        .args(["-sdk", "macosx", "metallib"])
        .args(air_files.iter().map(|f| f.to_str().unwrap()))
        .args(["-o", metallib_path.to_str().unwrap()]);

    let status = link_cmd
        .status()
        .expect("Failed to run xcrun metallib linker");

    if !status.success() {
        panic!("Metal shader library linking failed");
    }

    println!(
        "cargo::rustc-env=METAL_LIB_PATH={}",
        metallib_path.display()
    );

    // Compile os_signpost ObjC shim for Instruments integration.
    let signpost_src = Path::new(&manifest_dir).join("build/signpost.m");
    let signpost_obj = out_dir.join("signpost.o");
    println!("cargo:rerun-if-changed={}", signpost_src.display());
    let status = Command::new("clang")
        .args([
            "-c", "-O2", "-fobjc-arc",
            signpost_src.to_str().unwrap(),
            "-o", signpost_obj.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to compile signpost shim");
    assert!(status.success(), "signpost.m compilation failed");
    let signpost_lib = out_dir.join("libsignpost.a");
    Command::new("ar")
        .args(["rcs", signpost_lib.to_str().unwrap(), signpost_obj.to_str().unwrap()])
        .status()
        .expect("Failed to create signpost static library");
    println!("cargo::rustc-link-search=native={}", out_dir.display());
    println!("cargo::rustc-link-lib=static=signpost");
}
