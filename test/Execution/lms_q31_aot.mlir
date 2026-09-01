// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/lms_q31_aot.c %t.o -o %t
// RUN: %t
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" > %t.pipeline.mlir
// RUN: ondrix-translate %t.pipeline.mlir --mlir-to-llvmir > %t.pipeline.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.pipeline.ll -o %t.pipeline.o
// RUN: cc %S/Inputs/lms_q31_aot.c %t.pipeline.o -o %t.pipeline
// RUN: %t.pipeline

// Two tap counts derive two different product shifts, and K = 1 derives none.
// The quantized weight state is part of the contract, so a one-LSB difference
// in any update compounds through the recursion and the reference has to
// reproduce every boundary in order.

func.func @lms_k16_q31(%x: tensor<48xi32>, %d: tensor<48xi32>, %w: tensor<16xi32>)
    -> (tensor<48xi32>, tensor<16xi32>) attributes {llvm.emit_c_interface} {
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 134217728 : i64,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<48xi32>, tensor<48xi32>, tensor<16xi32>) -> (tensor<48xi32>, tensor<16xi32>)
  return %e, %a : tensor<48xi32>, tensor<16xi32>
}

func.func @lms_k5_q31_floor(%x: tensor<48xi32>, %d: tensor<48xi32>, %w: tensor<5xi32>)
    -> (tensor<48xi32>, tensor<5xi32>) attributes {llvm.emit_c_interface} {
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 134217728 : i64,
    product_rounding = #ondsp.rounding<toward_negative>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<48xi32>, tensor<48xi32>, tensor<5xi32>) -> (tensor<48xi32>, tensor<5xi32>)
  return %e, %a : tensor<48xi32>, tensor<5xi32>
}

func.func @lms_k1_q31(%x: tensor<48xi32>, %d: tensor<48xi32>, %w: tensor<1xi32>)
    -> (tensor<48xi32>, tensor<1xi32>) attributes {llvm.emit_c_interface} {
  %e, %a = ondrix.lms %x, %d, %w {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    step_size = 134217728 : i64,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<48xi32>, tensor<48xi32>, tensor<1xi32>) -> (tensor<48xi32>, tensor<1xi32>)
  return %e, %a : tensor<48xi32>, tensor<1xi32>
}
