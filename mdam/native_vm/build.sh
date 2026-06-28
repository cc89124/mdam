#!/usr/bin/env bash
# Build the native MDAM batch VM shared library (native_mdam_vm.so).
# Two translation units: this VM + the verified dense core kernel from clifft_axis.
# Produces a byte-identical 387944-byte .so on the reference toolchain (g++ 11.4).
set -e
cd "$(dirname "$0")"
g++ -O3 -march=native -std=c++17 -DNDEBUG -shared -fPIC \
    native_mdam_vm.cpp \
    ../clifft_axis/cpp/mdm_core_executor.cpp \
    -o native_mdam_vm.so
echo "built native_mdam_vm.so ($(stat -c%s native_mdam_vm.so) bytes)"
