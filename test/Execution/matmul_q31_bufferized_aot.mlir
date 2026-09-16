// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=preserve-bufferizable-reductions=true --empty-tensor-to-alloc-tensor "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-fixed-decimate-outputs=vector-width=8 --vectorize-ondsp-fixed-memref-reduce="vector-width=8 chunk-multiple=4" --normalize-ondsp-fixed-vector-reduce --convert-ondrix-static-results-to-out-params --forward-ondrix-result-buffers=distinct-out-params=true --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-scf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=VECTOR --implicit-check-not=ondsp. < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: opt -O3 -vectorize-slp=false -vectorize-loops=false -S %t.ll -o %t.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.opt.ll -o %t.o
// RUN: cc %S/Inputs/matmul_q31_aot.c %t.o -o %t
// RUN: %t

// Same kernels and same independent reference as matmul_q31_aot.mlir, reached
// through the bufferized reduce_mac instead of the tensor lowering: the
// requantized product and the wrapping i64 accumulator must reproduce the
// tensor form's double boundary bit for bit, at both derived shifts.

// VECTOR-LABEL: llvm.func @matmul_k64_q31
// VECTOR-LABEL: llvm.func @matmul_k5_q31_floor

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
