// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DERIVED
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128 accumulator-chains=8" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DECLARED
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0 accumulator-chains=8" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCALAR

// How many partial-sum chains cover the target's multiply-add latency is not
// predicted by its register width: the width-derived count is a host-class
// heuristic, and a target that mispredicts declares its own for every kernel.

// DERIVED: vectorize-ondsp-fp-fast-memref-reduce{{.*}}interleave=4
// DECLARED: vectorize-ondsp-fp-fast-memref-reduce{{.*}}interleave=8
// A core with no usable lanes still carries the declared chain count.
// SCALAR: vectorize-ondsp-fp-fast-memref-reduce{{.*}}interleave=8{{.*}}vector-width=1

func.func @dot(%lhs: memref<64xf32>, %rhs: memref<64xf32>, %init: f32) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<64xf32>, memref<64xf32>) -> f32
  return %r : f32
}
