// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/rms_q15_aot.c %t.o -o %t -lm
// RUN: %t

// Root-mean-square against an independent reference that computes the
// exact sum of squares, the nearest-even mean, and a correction-looped
// integer square root (a different algorithm than the compiled bit-by-bit
// form). Directed corpus: zero (rms 0), all-maximum, the all-minimum
// corner and the mixed 63-minimum-plus-one-maximum witness (root rounds
// up to 32768 before the clamp), single full-scale samples, and DC levels whose
// mean is an exact square.

func.func @rms64_q15(%input: tensor<64xi16>) -> tensor<1xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

func.func @rms2_q15(%input: tensor<2xi16>) -> tensor<1xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<2xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}

func.func @rms64_floor_q15(%input: tensor<64xi16>) -> tensor<1xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<64xi16>) -> tensor<1xi16>
  return %result : tensor<1xi16>
}
