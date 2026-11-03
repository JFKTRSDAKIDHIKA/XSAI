#!/bin/bash
# usage: run_cases.sh <output-name> <cases...> [-- extra framework args]
set -e
echo 1000 > /proc/self/oom_score_adj
source /das/ts01/home/jiashuao/AMENPU/stage-work/fa-xarch-001/xsai/env.sh
export RAMULATOR_PREFIX=$HOME/.local/opt/ramulator2/72427a1bba37
cd /das/ts01/home/jiashuao/AMENPU/wt-xarch-bench
out=$1; shift
cases=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do cases+=("$1"); shift; done; [ "$1" == "--" ] && shift
/das/ts01/home/jiashuao/AMENPU/stage-work/fa-e2e-chisel-sv-1019/venv/bin/python -m coralbench.framework.run \
  --benchmark attention --platform XSAI --xsai-emu /das/ts01/home/jiashuao/AMENPU/XSAI/build/verilator-compile/emu \
  --cases "${cases[@]}" --output output/$out --workers ${WORKERS:-3} --max-cycles 200000000 "$@"
