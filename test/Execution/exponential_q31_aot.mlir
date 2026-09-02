// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/exponential_q31_aot.c %t.o -lm -o %t
// RUN: %t

// The 2^32 domains cannot be swept in a test, so the gate proves two things:
// bit-exactness against an independent restatement of each declared program,
// and the one-LSB distance to the correctly rounded value the contracts claim.

func.func @log2_q0_32(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.log2 %input {
    numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}

func.func @exp2_q6_26(%input: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.exp2 %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 26>,
    output_numeric = #ondsp.fixed<unsigned, storage = i32, frac = 32>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi32>) -> tensor<4096xi32>
  return %result : tensor<4096xi32>
}
