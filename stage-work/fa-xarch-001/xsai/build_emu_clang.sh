#!/bin/bash
# XSAI emu: DefaultMatrixFp32Config (DefaultMatrixConfig + FP32 MMA), coralBench
# Ramulator2 DDR4 memory, performance counters kept (no RELEASE / DISABLE_PERF).
set -e
echo 1000 > /proc/self/oom_score_adj
source /das/ts01/home/jiashuao/AMENPU/stage-work/fa-xarch-001/xsai/env.sh
cd $NOOP_HOME
echo "START $(date -Is)"
git rev-parse HEAD; git submodule status
make emu CONFIG=DefaultMatrixFp32Config WITH_RAMULATOR2=1 EMU_THREADS=8 JVM_XMX=40G -j48 CXX=clang++ LINK=clang++
echo "DONE $(date -Is)"
ls -la build/emu
