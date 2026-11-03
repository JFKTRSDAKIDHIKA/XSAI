#!/bin/bash
set -e
echo 1000 > /proc/self/oom_score_adj
source /das/ts01/home/jiashuao/AMENPU/stage-work/fa-xarch-001/xsai/env.sh
cd $NOOP_HOME
mill -i CUTE.test.runMain cute.FReducePEFp32Emit /das/ts01/home/jiashuao/AMENPU/stage-work/fa-xarch-001/xsai/fpe-unit/rtl
