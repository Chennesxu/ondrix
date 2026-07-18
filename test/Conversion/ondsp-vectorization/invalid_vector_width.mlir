// RUN: not ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=0" 2>&1 | FileCheck %s
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=1" 2>&1 | FileCheck %s --check-prefix=PROVEN-WIDTH
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="max-elements=0" 2>&1 | FileCheck %s --check-prefix=PROVEN-LIMIT

// CHECK: error: vector-width must be positive
// PROVEN-WIDTH: error: vector-width must be greater than one
// PROVEN-LIMIT: error: max-elements must be positive

module {
}
