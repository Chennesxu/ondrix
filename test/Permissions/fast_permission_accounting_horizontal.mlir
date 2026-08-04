// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8" --mlir-print-op-generic | FileCheck %s --check-prefix=SEPARATE
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8 supports-vector-fma=true" --mlir-print-op-generic | FileCheck %s --check-prefix=FUSED

// The horizontal route spends R on the cross-lane fold. Under the default term
// selection its lane accumulation is a rounded product and an addition, so F
// is not spent; declaring a vector FMA moves the same route to {R, F}. Read
// generically because vector.reduction has no attr-dict in its custom form.

// SEPARATE: ondsp.fast_used = "reassociate_reduction_terms"
// SEPARATE-NOT: ondsp.fast_used = "fuse_multiply_add"
// FUSED-DAG: ondsp.fast_used = "reassociate_reduction_terms"
// FUSED-DAG: ondsp.fast_used = "fuse_multiply_add"
func.func @horizontal_route(%init: f32, %lhs: memref<32xf32>, %rhs: memref<32xf32>) -> f32 {
  %result = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<32xf32>, memref<32xf32>) -> f32
  return %result : f32
}
