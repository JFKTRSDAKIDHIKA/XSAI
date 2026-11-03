# XSAI build/run environment for the fa-xarch study.
export E2E=/das/ts01/home/jiashuao/e2e
export NOOP_HOME=/das/ts01/home/jiashuao/AMENPU/XSAI
export PATH=$E2E/tools/verilator536/bin:$E2E/tools/bin:$PATH
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export VERILATOR_ROOT=$E2E/tools/verilator536/share/verilator
export MILL_HOME=$E2E/.mill
export COURSIER_REPOSITORIES="https://maven.aliyun.com/repository/central|central"
export CORALBENCH_DRAM_DIR=/das/ts01/home/jiashuao/AMENPU/wt-xarch-bench/platform/dram
export CORALBENCH_RAMULATOR_LIB=$HOME/.local/opt/ramulator2/72427a1bba37/libameramulator.so
export CORALBENCH_RAMULATOR_CONFIG=$HOME/.local/opt/ramulator2/72427a1bba37/ddr4-3200-1ch.yaml
# sqlite3 headers/library for chisel_db/perfCCT (no system libsqlite3-dev on this host)
export CPATH=$E2E/tools/shim/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$E2E/tools/shim/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$E2E/tools/shim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# program builders (coralBench main df0bf53+): toolchain and Chipyard locations
export XARCH_RISCV_BIN=/das/ts01/home/jiashuao/e2e/tools/riscv/bin
export XARCH_CHIPYARD=/das/ts01/home/jiashuao/e2e/chipyard
export RAMULATOR_PROFILE=500mhz
