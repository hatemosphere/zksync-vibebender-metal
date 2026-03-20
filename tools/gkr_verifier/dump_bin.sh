#!/bin/sh
set -euo pipefail

CIRCUITS="add_sub jump_branch_slt shift_binop"

for circuit in $CIRCUITS; do
    echo "==> Building RISC-V binary for $circuit"

    rm -f app_${circuit}.bin app_${circuit}.elf app_${circuit}.text

    cargo build --release --features "$circuit" -Z panic-immediate-abort -Z build-std=core,alloc
    cargo objcopy --release --features "$circuit" -Z panic-immediate-abort -Z build-std=core,alloc -- -O binary ${circuit}.bin
    cargo objcopy --release --features "$circuit" -Z panic-immediate-abort -Z build-std=core,alloc -- -R .text ${circuit}.elf
    cargo objcopy --release --features "$circuit" -Z panic-immediate-abort -Z build-std=core,alloc -- -O binary --only-section=.text ${circuit}.text
done

echo "==> All binaries built"
