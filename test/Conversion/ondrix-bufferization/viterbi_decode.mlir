// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" | FileCheck %s

// The decoder bufferizes to its exact wide realization; a declared symbol
// bound reaches the symbol buffer as the assumption a certificate reads.
// CHECK-LABEL: func.func @bounded
// CHECK: %[[BITS:.*]] = memref.alloc() {{.*}} : memref<32xi8>
// CHECK: %[[DECLARED:.*]] = ondsp.assume_magnitude_bound %arg0 {bound = 127 : i64} : memref<512xi16>
// CHECK: ondsp.viterbi_decode %[[DECLARED]], %[[BITS]] {constraint_length = 7 : i64, metric_bits = 32 : i64, polynomials = array<i64: 121, 91>, renormalization_period = 0 : i64, symbol_bound = 32768 : i64, unreachable_metric = -1073741825 : i64}
// CHECK-LABEL: func.func @undeclared
// CHECK-NOT: ondsp.assume_magnitude_bound
// CHECK: ondsp.viterbi_decode %arg0
func.func @bounded(%symbols: tensor<512xi16>) -> tensor<32xi8> {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 7 : i64,
      polynomials = array<i64: 121, 91>, symbol_bound = 127 : i64}
      : (tensor<512xi16>) -> tensor<32xi8>
  return %bits : tensor<32xi8>
}
func.func @undeclared(%symbols: tensor<32xi16>) -> tensor<2xi8> {
  %bits = ondrix.viterbi_decode %symbols {constraint_length = 3 : i64,
      polynomials = array<i64: 7, 5>} : (tensor<32xi16>) -> tensor<2xi8>
  return %bits : tensor<2xi8>
}
