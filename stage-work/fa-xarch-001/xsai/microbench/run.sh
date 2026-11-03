#!/bin/bash
# usage: run.sh <image.bin> <log>
echo 1000 > /proc/self/oom_score_adj
source /das/ts01/home/jiashuao/AMENPU/stage-work/fa-xarch-001/xsai/env.sh
R=$HOME/.local/opt/ramulator2/72427a1bba37
export CORALBENCH_RAMULATOR_LIB=$R/libameramulator.so CORALBENCH_RAMULATOR_CONFIG=$R/ddr4-3200-1ch.yaml LD_LIBRARY_PATH=$R:$LD_LIBRARY_PATH
cd "$(dirname "$2")"
/das/ts01/home/jiashuao/AMENPU/XSAI/build/verilator-compile/emu -i "$1" --no-diff -C 20000000 > "$2" 2>&1
echo "EXIT $?" >> "$2"
