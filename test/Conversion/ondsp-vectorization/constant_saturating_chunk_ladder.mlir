// RUN: ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 chunk-multiple=4 max-elements=64 proof-trace-output=%t.json" | FileCheck %s
// RUN: FileCheck %s --check-prefix=TRACE --input-file=%t.json
// RUN: ondrix-opt %s --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.json max-elements=64" | FileCheck %s --check-prefix=REPLAY
// RUN: ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 chunk-multiple=1 max-elements=64 proof-trace-output=%t.narrow.json" > /dev/null
// RUN: sed 's/"chunk_multiple":1/"chunk_multiple":4/' %t.narrow.json > %t.forged.json
// RUN: not ondrix-opt %s --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.forged.json max-elements=64" 2>&1 | FileCheck %s --check-prefix=FORGED

memref.global "private" constant @sixteen : memref<16xi16> =
  dense<[1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]>
memref.global "private" constant @eight : memref<8xi16> =
  dense<[1, -2, 3, -4, 5, -6, 7, -8]>

// Sixteen coefficients admit the widest rung, so one certified chunk spans
// four machine vectors and the loop collapses once instead of four times.
func.func @wide_chunk(%input: memref<16xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %coefficients = memref.get_global @sixteen : memref<16xi16>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.reduce_mac %zero, %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       memref<16xi16>, memref<16xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @wide_chunk
// CHECK: vector.load {{.*}} : memref<16xi16>, vector<16xi16>
// CHECK: vector.reduction <add>, {{.*}} : vector<16xi64> into i64
// CHECK-NOT: ondsp.reduce_mac

// Eight coefficients cannot fill the widest rung; the ladder steps down rather
// than losing vectorization, which is what makes the fallback load-bearing.
func.func @narrow_chunk(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %coefficients = memref.get_global @eight : memref<8xi16>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.reduce_mac %zero, %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @narrow_chunk
// CHECK: vector.load {{.*}} : memref<8xi16>, vector<8xi16>
// CHECK: vector.reduction <add>, {{.*}} : vector<8xi64> into i64
// CHECK-NOT: ondsp.reduce_mac

// The width is per subject in the record, not per module.
// TRACE-DAG: "chunk_multiple":{{ *}}4
// TRACE-DAG: "chunk_width":{{ *}}16
// TRACE-DAG: "chunk_width":{{ *}}8

// REPLAY-LABEL: func.func @wide_chunk

// FORGED: error: proof trace record names a chunk width the selection would not choose
