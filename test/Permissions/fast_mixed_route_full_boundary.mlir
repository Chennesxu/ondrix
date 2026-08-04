// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8" --mlir-print-op-generic > %t.sep.mlir
// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fp-fast-memref-reduce="vector-width=8 supports-vector-fma=true" --mlir-print-op-generic > %t.fma.mlir
// RUN: FileCheck %s --check-prefix=ROUTE < %t.sep.mlir
// RUN: FileCheck %s --check-prefix=ROUTE < %t.fma.mlir
// RUN: FileCheck %s --check-prefix=RECORD-R < %t.sep.mlir
// RUN: FileCheck %s --check-prefix=RECORD-R < %t.fma.mlir
// RUN: FileCheck %s --check-prefix=SEP-TOTAL < %t.sep.mlir
// RUN: FileCheck %s --check-prefix=SEP-EDGE < %t.sep.mlir
// RUN: FileCheck %s --check-prefix=FMA-TOTAL < %t.fma.mlir
// RUN: FileCheck %s --check-prefix=FMA-LANE < %t.fma.mlir
// RUN: FileCheck %s --check-prefix=UNION < %t.sep.mlir
// RUN: FileCheck %s --check-prefix=UNION < %t.fma.mlir

// One declaration, two mechanisms in one function; the argument is in
// docs/f32-contract-evidence.md.

// Mechanism cardinality, not site identity.
// ROUTE-COUNT-1: "vector.reduction"
// ROUTE-NOT: "vector.reduction"

// RECORD-R: "vector.reduction"{{.*}}ondsp.fast_used = ["rebuild_reduction_tree"]

// Both fused events are scalar edge events; a compensating pair keeps the count.
// SEP-TOTAL-COUNT-2: "math.fma"
// SEP-TOTAL-NOT: "math.fma"
// SEP-EDGE-COUNT-2: "math.fma"{{.*}}ondsp.fast_used = ["fuse_multiply_add"]{{.*}}(f32, f32, f32) -> f32
// SEP-EDGE-NOT: "math.fma"

// A declared vector FMA adds F to the interior and leaves the edges alone.
// FMA-TOTAL-COUNT-5: "math.fma"
// FMA-TOTAL-NOT: "math.fma"
// FMA-LANE-COUNT-1: "math.fma"{{.*}}ondsp.fast_used = ["fuse_multiply_add"]{{.*}}(vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// FMA-LANE-NOT: "math.fma"{{.*}}vector<8xf32>

// The set is the compilation's, identical across two site distributions.
// UNION: ondsp.fast_used = ["fuse_multiply_add", "rebuild_reduction_tree"]

func.func @full_boundary_mixed_route(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<full>,
    numeric = #ondsp.fp<format = f32, contract = fast>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %result : tensor<?xf32>
}
