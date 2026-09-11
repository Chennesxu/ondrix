// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128 widening-multiply-low-halves=false multiply-add-adjacent-pairs=false" | FileCheck %s

// The filter output is consumed inside the kernel, so it is a small local
// temporary: it lives on the stack, and only the returned state stays on the heap.
// CHECK-LABEL: llvm.func @sos_hash
// CHECK: llvm.alloca
// CHECK-NOT: llvm.call @free
func.func @sos_hash(%input: tensor<128xi16>, %coefficients: tensor<1x5xi16>,
                    %scales: tensor<1xi16>, %state: tensor<1x2xi16>) -> (i64, tensor<1x2xi16>) {
  %output, %next_state = ondrix.sos_filter_df2_fixed %input, %coefficients, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<toward_negative>,
    product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<toward_negative>
  } : (tensor<128xi16>, tensor<1x5xi16>, tensor<1xi16>, tensor<1x2xi16>)
      -> (tensor<128xi16>, tensor<1x2xi16>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %seed = arith.constant 1469598103934665603 : i64
  %prime = arith.constant 1099511628211 : i64
  %hash = scf.for %index = %c0 to %c128 step %c1 iter_args(%acc = %seed) -> (i64) {
    %value = tensor.extract %output[%index] : tensor<128xi16>
    %wide = arith.extui %value : i16 to i64
    %scaled = arith.muli %acc, %prime : i64
    %updated = arith.xori %scaled, %wide : i64
    scf.yield %updated : i64
  }
  return %hash, %next_state : i64, tensor<1x2xi16>
}
