// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8" --mlir-print-op-generic | FileCheck %s --check-prefix=SEPARATE
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8 supports-vector-fma=true" --mlir-print-op-generic | FileCheck %s --check-prefix=FUSED

// The horizontal route spends R on the cross-lane fold. Under the default term
// selection its lane accumulation is a rounded product and an addition, so F
// is not spent; declaring a vector FMA moves the same route to {R, F}. Read
// generically because vector.reduction has no attr-dict in its custom form.

// The static extent takes the batched branch unconditionally, so one case is
// generated and its permission set is the whole answer for this site.
// SEPARATE: ondsp.fast_selection = [{{[{]}}{{.*}}mechanism = "horizontal_separate"{{.*}}used_permissions = ["rebuild_reduction_tree"]{{.*}}when = ""
// SEPARATE-NOT: fuse_multiply_add
// FUSED: ondsp.fast_selection = [{{[{]}}{{.*}}mechanism = "horizontal_fused"{{.*}}used_permissions = ["fuse_multiply_add", "rebuild_reduction_tree"]
func.func @horizontal_route(%init: f32, %lhs: memref<32xf32>, %rhs: memref<32xf32>) -> f32 {
  %result = ondsp.reduce_mac %init, %lhs, %rhs {
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (f32, memref<32xf32>, memref<32xf32>) -> f32
  return %result : f32
}
