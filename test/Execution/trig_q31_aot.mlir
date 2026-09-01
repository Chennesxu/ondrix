// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/trig_q31_aot.c %t.o -o %t -lm
// RUN: %t

// The 2^32 phase domain cannot be swept in a test, so the gate proves two
// different things instead. Against an independent C restatement of the
// declared contract it must be bit-exact, and against libm it must stay
// within the two LSB the contract claims -- a bound the widened Q15 shape
// would miss by four orders of magnitude.

func.func @sine4096_q31(%phase: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.sine %phase {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @cosine4096_q31(%phase: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.cosine %phase {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}
