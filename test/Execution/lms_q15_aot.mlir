// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/lms_q15_aot.c %t.o -o %t
// RUN: %t

// Quantized-state LMS against an independent per-step reference: the
// error signal AND the final weights must match bit-exactly, so any
// deviation in any of the 2 + K rounding boundaries of any step is
// caught through the feedback. The harness also pins the family-9
// compounding witness: replacing the two-stage update quantization
// q15(q15(mu*e) * x) with the single-rounding fused product
// q15_30(mu*e*x) — a valid real-arithmetic reassociation — first
// diverges at sample 4 of the deterministic corpus and then compounds
// through the recursion to 172 of 256 diverging error samples and 6 of
// 8 diverging final weights.

func.func @lms8_q15(%x: tensor<256xi16>, %d: tensor<256xi16>, %w: tensor<8xi16>)
    -> (tensor<256xi16>, tensor<8xi16>) attributes {llvm.emit_c_interface} {
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 4096 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<256xi16>, tensor<256xi16>, tensor<8xi16>) -> (tensor<256xi16>, tensor<8xi16>)
  return %e, %wf : tensor<256xi16>, tensor<8xi16>
}

func.func @lms1_q15(%x: tensor<32xi16>, %d: tensor<32xi16>, %w: tensor<1xi16>)
    -> (tensor<32xi16>, tensor<1xi16>) attributes {llvm.emit_c_interface} {
  %e, %wf = ondrix.lms %x, %d, %w {
    step_size = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<32xi16>, tensor<32xi16>, tensor<1xi16>) -> (tensor<32xi16>, tensor<1xi16>)
  return %e, %wf : tensor<32xi16>, tensor<1xi16>
}
