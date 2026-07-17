// RUN: not ondrix-opt %s --specialize-ondrix-constant-fir="max-taps=0" 2>&1 | FileCheck %s
// RUN: not ondrix-opt %s --specialize-ondrix-constant-fir="max-taps=-1" 2>&1 | FileCheck %s

// CHECK: error: max-taps must be positive
module {
}
