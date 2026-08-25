// RUN: not ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=0" 2>&1 | FileCheck %s
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=1" 2>&1 | FileCheck %s --check-prefix=PROVEN-WIDTH
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="max-elements=0" 2>&1 | FileCheck %s --check-prefix=PROVEN-LIMIT

// A width the replay pass would refuse must be refused by the emitter too, or
// this compiler writes evidence it cannot revalidate.
// RUN: not ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4 chunk-multiple=0" 2>&1 | FileCheck %s --check-prefix=LADDER
// RUN: not ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4 chunk-multiple=-1" 2>&1 | FileCheck %s --check-prefix=LADDER
// RUN: not ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4 chunk-multiple=17" 2>&1 | FileCheck %s --check-prefix=LADDER
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 chunk-multiple=17 proof-trace-output=%t.refused.json" 2>&1 | FileCheck %s --check-prefix=LADDER
// RUN: not test -f %t.refused.json
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 chunk-multiple=4611686018427387904" 2>&1 | FileCheck %s --check-prefix=LADDER
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=8192 chunk-multiple=1" 2>&1 | FileCheck %s --check-prefix=LANES

// CHECK: error: vector-width must be positive
// PROVEN-WIDTH: error: vector-width must be greater than one
// PROVEN-LIMIT: error: max-elements must be positive
// LADDER: error: chunk-multiple must be between 1 and 16
// LANES: error: vector-width exceeds the 4096 lane limit

module {
}
