#!/bin/sh
set -euo pipefail

rm -f app.bin app.elf app.text

cargo build --release -Z panic-immediate-abort -Z build-std=core,alloc
cargo objcopy --release -Z panic-immediate-abort -Z build-std=core,alloc -- -O binary app.bin
cargo objcopy --release -Z panic-immediate-abort -Z build-std=core,alloc -- -R .text app.elf
cargo objcopy --release -Z panic-immediate-abort -Z build-std=core,alloc -- -O binary --only-section=.text app.text
