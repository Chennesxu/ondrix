// RUN: ondrix-opt %s --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --normalize-ondsp-fixed-vector-reduce --lower-ondsp-f32-reduce-to-scalar --canonicalize > %t.vector.mlir
// RUN: FileCheck %s --check-prefix=VECTOR < %t.vector.mlir
// RUN: ondrix-opt %t.vector.mlir --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/fir_filter_full_tensor_aot.c %t.o -lm -o %t
// RUN: %t
// RUN: cc %S/Inputs/fir_filter_full_tensor_mismatch.c %t.o -lm -o %t.mismatch
// RUN: not --crash %t.mismatch input
// RUN: not --crash %t.mismatch coefficients
// RUN: not --crash %t.mismatch output
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.generic.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.generic.mlir
// RUN: ondrix-translate %t.generic.mlir --mlir-to-llvmir > %t.generic.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.generic.ll -o %t.generic.o
// RUN: cc %S/Inputs/fir_filter_full_tensor_aot.c %t.generic.o -lm -o %t.generic
// RUN: %t.generic
// RUN: cc %S/Inputs/fir_filter_full_tensor_mismatch.c %t.generic.o -lm -o %t.generic.mismatch
// RUN: not --crash %t.generic.mismatch input
// RUN: not --crash %t.generic.mismatch coefficients
// RUN: not --crash %t.generic.mismatch output

// VECTOR-LABEL: func.func @q15_full_filter_value
// VECTOR-COUNT-3: cf.assert
// VECTOR-NOT: memref.alloc
// VECTOR-NOT: memref.copy
// VECTOR: scf.if
// VECTOR: ondsp.mac
// VECTOR-COUNT-2: vector.load {{.*}}vector<4xi16>
// VECTOR: arith.muli {{.*}} : vector<4xi32>
// VECTOR: scf.if
// VECTOR: ondsp.mac
// VECTOR-NOT: memref.alloc
// VECTOR-NOT: memref.copy
// VECTOR-LABEL: func.func @q31_full_filter_value
// VECTOR-COUNT-3: cf.assert
// VECTOR-NOT: memref.alloc
// VECTOR-NOT: memref.copy
// VECTOR: scf.if
// VECTOR-COUNT-2: vector.load {{.*}}vector<4xi32>
// VECTOR: arith.muli {{.*}} : vector<4xi64>
// VECTOR: scf.if
// VECTOR-NOT: memref.alloc
// VECTOR-NOT: memref.copy
// VECTOR-LABEL: func.func @q15_full_filter_short_input
// VECTOR: scf.if
// VECTOR: ondsp.mac
// VECTOR-LABEL: func.func @f32_full_filter_value
// VECTOR: scf.if
// VECTOR: math.fma
// VECTOR-LABEL: func.func @q15_full_shared_coeff_init
// VECTOR: memref.alloc
// VECTOR-NOT: memref.copy
// VECTOR: scf.for
// VECTOR-NOT: memref.alloc
// VECTOR-NOT: memref.copy
// VECTOR-LABEL: func.func @q15_full_shared_input_init
// VECTOR: memref.alloc
// VECTOR-NOT: memref.copy
// VECTOR: scf.for
// VECTOR-NOT: memref.alloc
// VECTOR-NOT: memref.copy

// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
// LOWERED-NOT: tensor.
// LOWERED-NOT: @memrefCopy

func.func @q15_full_filter_value(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>,
    %index: index) -> i16 {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  %value = tensor.extract %result[%index] : tensor<?xi16>
  return %value : i16
}

func.func @q31_full_filter_value(
    %input: tensor<?xi32>, %coeffs: tensor<?xi32>, %init: tensor<?xi32>,
    %index: index) -> i32 {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %value = tensor.extract %result[%index] : tensor<?xi32>
  return %value : i32
}

func.func @q15_full_filter_short_input(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>,
    %index: index) -> i16 {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  %value = tensor.extract %result[%index] : tensor<?xi16>
  return %value : i16
}

func.func @f32_full_filter_value(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>,
    %index: index) -> f32 {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<full>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %value = tensor.extract %result[%index] : tensor<?xf32>
  return %value : f32
}

func.func @q15_full_shared_coeff_init(
    %input: tensor<?xi16>, %coeffs_and_init: tensor<?xi16>, %index: index)
    -> i16 {
  %result = ondrix.fir_filter %input, %coeffs_and_init, %coeffs_and_init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  %value = tensor.extract %result[%index] : tensor<?xi16>
  return %value : i16
}

func.func @q15_full_shared_input_init(
    %input_and_init: tensor<?xi16>, %coeffs: tensor<?xi16>, %index: index)
    -> i16 {
  %result = ondrix.fir_filter %input_and_init, %coeffs, %input_and_init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<full>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  %value = tensor.extract %result[%index] : tensor<?xi16>
  return %value : i16
}
