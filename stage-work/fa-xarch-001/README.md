# stage-work/fa-xarch-001 — XSAI FP32 matrix path and FA benchmark scratch

Snapshot of the working notes for the XSAI FP32 (`DefaultMatrixFp32Config`) work
and the FlashAttention cross-platform benchmark (2026-10-20 .. 2026-10-22).

## Contents

- `xsai/MATRIX_ISA_GUIDE.md` — XSAI CUTE matrix + RVV programmer's guide for a
  bare-metal FP32 FlashAttention kernel (encodings, tile geometry, `mcfg`
  type codes, numerics, suggested block schedule).
- `xsai/build_emu.sh`, `xsai/build_emu_clang.sh`, `xsai/emit_fpe.sh`,
  `xsai/env.sh` — emulator / FPE emission scripts. `env.sh` still contains the
  absolute paths of the original machine (`$E2E`, `$NOOP_HOME`, Ramulator2
  install); edit them before use.
- `xsai/fpe-unit/` — bit-exact FReducePE unit test: `tb_freducepe.cpp`
  (testbench), `rtl/FReducePE.sv` (emitted DUT), `build.log` (Verilator build),
  `run-20000.log` (20 000 vectors × {FP32, TF32, BF16, INT8} against the
  difftest CUTE MMA model). The `obj/` build directory (137 MB) was not
  committed; re-run the build to regenerate it.
- `xsai/logs-*.log` — run logs for microbenchmarks, phase sweeps and the
  emulator builds.
- `PROGRESS.txt` — day-by-day progress log of the whole cross-platform study
  (also covers Gemmini and AMEOOO; those parts are not vendored here).

## Where the code lives

Everything the FP32 path needs is vendored into this repository (no git
submodules): XSAI core in `src/`, the CUTE accelerator in `CUTE/`,
the QIF/FPE/TMAC in `CUTE/cute-fpe/`, and the bit-exact CUTE MMA reference
model plus the Ramulator2 emulator backend in `difftest/`.
