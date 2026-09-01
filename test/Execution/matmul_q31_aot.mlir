// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/matmul_q31_aot.c %t.o -o %t
// RUN: %t
// The canonical pipeline is a separate route and has to be compiled too.
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.pipeline.mlir
// RUN: ondrix-translate %t.pipeline.mlir --mlir-to-llvmir > %t.pipeline.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pipeline.ll -o %t.pipeline.o
// RUN: cc %S/Inputs/matmul_q31_aot.c %t.pipeline.o -o %t.pipeline
// RUN: %t.pipeline

// Two inner extents carry two different derived product shifts. The two Q15
// arms differ only in the declared export rounding and must NOT agree: before
// the tensor lowering was fixed it gave both of them nearest_even.

func.func @matmul_k64_q31(%a: tensor<3x64xi32>, %b: tensor<64x3xi32>) -> tensor<3x3xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<3x64xi32>, tensor<64x3xi32>) -> tensor<3x3xi32>
  return %result : tensor<3x3xi32>
}

func.func @matmul_k5_q31_floor(%a: tensor<3x5xi32>, %b: tensor<5x3xi32>) -> tensor<3x3xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<3x5xi32>, tensor<5x3xi32>) -> tensor<3x3xi32>
  return %result : tensor<3x3xi32>
}

func.func @matmul_k8_q15_even(%a: tensor<3x8xi16>, %b: tensor<8x3xi16>) -> tensor<3x3xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<3x8xi16>, tensor<8x3xi16>) -> tensor<3x3xi16>
  return %result : tensor<3x3xi16>
}

func.func @matmul_k8_q15_floor(%a: tensor<3x8xi16>, %b: tensor<8x3xi16>) -> tensor<3x3xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.matmul %a, %b {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<3x8xi16>, tensor<8x3xi16>) -> tensor<3x3xi16>
  return %result : tensor<3x3xi16>
}
