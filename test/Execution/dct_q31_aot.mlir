// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --canonicalize --cse --convert-ondrix-static-results-to-out-params --forward-ondrix-result-buffers=distinct-out-params=true --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/dct_q31_aot.c %t.o -o %t -lm
// RUN: %t
// The canonical pipeline routes Q31 away from the reduce_mac bufferization,
// so it is a separate object and has to be executed too.
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.pipeline.mlir
// RUN: ondrix-translate %t.pipeline.mlir --mlir-to-llvmir > %t.pipeline.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pipeline.ll -o %t.pipeline.o
// RUN: cc %S/Inputs/dct_q31_aot.c %t.pipeline.o -o %t.pipeline -lm
// RUN: %t.pipeline

// Two Q31 extents derive two different product shifts, and the two N = 8 arms
// differ only in the declared product rounding: the harness REQUIRES them to
// disagree, which a lowering that pinned one mode would fail.

func.func @dct8_q31(%input: tensor<8xi32>) -> tensor<8xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

func.func @dct8_q31_floor(%input: tensor<8xi32>) -> tensor<8xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}

func.func @dct64_q31(%input: tensor<64xi32>) -> tensor<64xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 24>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<64xi32>
  return %result : tensor<64xi32>
}

func.func @dct8_q15(%input: tensor<8xi16>) -> tensor<8xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_numeric = #ondsp.fixed<signed, storage = i16, frac = 11>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi16>) -> tensor<8xi16>
  return %result : tensor<8xi16>
}
