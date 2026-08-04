// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/exponential_q15_aot.c %t.o -lm -o %t
// RUN: %t

// Both input domains are 2^16 values, so both are swept whole against a
// reference that regenerates the declared table from its definition rather
// than reading the compiler's constant. The harness also measures the
// log2/exp2 round trip instead of asserting it is the identity, which it is
// not and the contract never claimed.

func.func @log2_q0_16(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.log2 %input {
    numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @exp2_q5_11(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.exp2 %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    output_numeric = #ondsp.fixed<unsigned, storage = i16, frac = 16>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}
