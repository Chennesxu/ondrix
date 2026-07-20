// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=8 proof-trace-output=%t.json" 2>&1 | FileCheck %s
// RUN: not ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=65537 proof-trace-output=%t.json" 2>&1 | FileCheck %s --check-prefix=LIMIT

module {
}

// CHECK: error: proof trace requested, but no reduction was proof-authorized
// LIMIT: error: proof trace max-elements exceeds the experimental audit limit of 65536
