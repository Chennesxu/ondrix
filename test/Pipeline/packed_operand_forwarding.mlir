// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128 widening-multiply-low-halves=false multiply-add-adjacent-pairs=false" | FileCheck %s --check-prefix=ORDERED
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" | FileCheck %s --check-prefix=BATCHED

// At 128 bits the column batcher declines the requantized Q31 product and the
// transposed copy of B would only feed ordered reductions: the copy is gone
// and one allocation (the result) remains. At 256 the batcher consumes it.
// ORDERED-LABEL: llvm.func @matmul4x16x3_q31
// ORDERED-COUNT-1: llvm.call @malloc
// ORDERED-NOT: llvm.call @malloc
// ORDERED-NOT: llvm.call @free
// BATCHED-LABEL: llvm.func @matmul4x16x3_q31
// BATCHED-COUNT-1: llvm.call @malloc
// BATCHED-NOT: llvm.call @malloc
func.func @matmul4x16x3_q31(%a: tensor<4x16xi32>, %b: tensor<16x3xi32>) -> tensor<4x3xi32>
    attributes {llvm.emit_c_interface} {
  %c = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4x16xi32>, tensor<16x3xi32>) -> tensor<4x3xi32>
  return %c : tensor<4x3xi32>
}
