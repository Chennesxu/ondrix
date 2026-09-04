// RUN: ondrix-opt %s --convert-ondrix-to-ondsp=preserve-bufferizable-reductions=true --empty-tensor-to-alloc-tensor "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --cse --canonicalize --vectorize-ondsp-fixed-elementwise-loops=vector-width=8 --vectorize-ondsp-fixed-memref-reduce="vector-width=8 chunk-multiple=4" --normalize-ondsp-fixed-vector-reduce --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-scf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=VECTOR --implicit-check-not=ondsp. < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: opt -O3 -vectorize-slp=false -vectorize-loops=false -S %t.ll -o %t.opt.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.opt.ll -o %t.o
// RUN: cc %S/Inputs/rms_q31_aot.c %t.o -o %t -lm
// RUN: %t

// Same kernels and same independent reference as rms_q31_aot.mlir, reached
// through the bufferized reduce_mac instead of the tensor lowering: the
// pre-requantized scratch copy, the exact-modulo lane sums, and the restoring
// left shift must reproduce the tensor form bit for bit, at both derived
// pre-shifts.

// VECTOR-LABEL: llvm.func @rms16_q31
// VECTOR-LABEL: llvm.func @rms64_q31

func.func @rms16_q31(%input: tensor<16xi32>) -> tensor<1xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<16xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

func.func @rms64_q31(%input: tensor<64xi32>) -> tensor<1xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<64xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}

func.func @rms64_q31_floor(%input: tensor<64xi32>) -> tensor<1xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.rms %input {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    input_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<toward_negative>
  } : (tensor<64xi32>) -> tensor<1xi32>
  return %result : tensor<1xi32>
}
