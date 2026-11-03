#!/bin/bash
set -e
cd "$(dirname "$0")"
T=/das/ts01/home/jiashuao/AMENPU/wt-xarch-bench/platform/xsai/target
X=/das/ts01/home/jiashuao/AMENPU/wt-xarch-bench/benchmark/kernels/attention/xarch
B=/das/ts01/home/jiashuao/e2e/tools/riscv/bin
$B/riscv64-unknown-elf-gcc -march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicond_zicbom_zicbop -mabi=lp64d -mcmodel=medany -std=gnu11 -O2 \
  -ffreestanding -fno-builtin -nostdlib -nostartfiles -static -I. -I$X -T $T/link.ld $T/crt.S $T/runtime.c mb.c -lgcc -o mb.elf
$B/riscv64-unknown-elf-objcopy -O binary mb.elf mb.bin
$B/riscv64-unknown-elf-objdump -d mb.elf > mb.dis
