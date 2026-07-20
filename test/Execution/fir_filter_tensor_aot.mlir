// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: FileCheck %s --check-prefix=ALIAS < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/fir_filter_tensor_aot.c %t.o -lm -o %t
// RUN: %t
// RUN: cc %S/Inputs/fir_filter_tensor_mismatch.c %t.o -lm -o %t.mismatch
// RUN: not --crash %t.mismatch empty
// RUN: not --crash %t.mismatch short
// RUN: not --crash %t.mismatch output
// RUN: ondrix-opt %s --tile-ondrix-fir-filter="tile-size=4" --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=64" --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --normalize-ondsp-fixed-vector-reduce --lower-ondsp-f32-reduce-to-scalar --canonicalize > %t.tiled-vector.mlir
// RUN: FileCheck %s --check-prefix=TILED-VECTOR < %t.tiled-vector.mlir
// RUN: ondrix-opt %t.tiled-vector.mlir --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.tiled.mlir
// RUN: FileCheck %s --check-prefix=TILED-LOWERED < %t.tiled.mlir
// RUN: ondrix-translate %t.tiled.mlir --mlir-to-llvmir > %t.tiled.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.tiled.ll -o %t.tiled.o
// RUN: cc %S/Inputs/fir_filter_tensor_aot.c %t.tiled.o -lm -o %t.tiled
// RUN: %t.tiled

// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
// LOWERED-NOT: tensor.

// ALIAS-LABEL: llvm.func @q15_fir_filter_shared_coeff_init
// ALIAS: llvm.call @malloc
// ALIAS: "llvm.intr.memcpy"

// TILED-VECTOR-LABEL: func.func @q15_fir_filter_value
// TILED-VECTOR-COUNT-3: cf.assert
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR: scf.for
// TILED-VECTOR-NOT: cf.assert
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR-COUNT-2: vector.load {{.*}}vector<4xi16>
// TILED-VECTOR: arith.muli {{.*}} : vector<4xi32>
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR-LABEL: func.func @q31_fir_filter_value
// TILED-VECTOR-COUNT-3: cf.assert
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR: scf.for
// TILED-VECTOR-NOT: cf.assert
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR-COUNT-2: vector.load {{.*}}vector<4xi32>
// TILED-VECTOR: arith.muli {{.*}} : vector<4xi64>
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR-LABEL: func.func @q15_fir_filter_shared_coeff_init
// TILED-VECTOR-COUNT-3: cf.assert
// TILED-VECTOR: memref.alloc
// TILED-VECTOR: memref.copy
// TILED-VECTOR: scf.for
// TILED-VECTOR-NOT: cf.assert
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR-COUNT-2: vector.load {{.*}}vector<4xi16>
// TILED-VECTOR: arith.muli {{.*}} : vector<4xi32>
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR-LABEL: func.func @f32_fir_filter_value
// TILED-VECTOR-COUNT-3: cf.assert
// TILED-VECTOR: scf.for
// TILED-VECTOR-NOT: cf.assert
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy
// TILED-VECTOR: math.fma
// TILED-VECTOR-NOT: memref.alloc
// TILED-VECTOR-NOT: memref.copy

// TILED-VECTOR-LABEL: func.func @q15_proven_fir_filter_value
// TILED-VECTOR: %[[Q15_SUM:.*]] = vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// TILED-VECTOR: ondsp.acc_add_term {{.*}}, %[[Q15_SUM]]
// TILED-VECTOR-NOT: vector.extract
// TILED-VECTOR-NOT: ondsp.reduce_mac
// TILED-VECTOR-LABEL: func.func @q31_proven_fir_filter_value
// TILED-VECTOR: %[[Q31_SUM:.*]] = vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// TILED-VECTOR: ondsp.acc_add_term {{.*}}, %[[Q31_SUM]]
// TILED-VECTOR-NOT: vector.extract
// TILED-VECTOR-NOT: ondsp.reduce_mac

// TILED-LOWERED-NOT: ondrix.
// TILED-LOWERED-NOT: ondsp.
// TILED-LOWERED-NOT: tensor.
// TILED-LOWERED-NOT: affine.
// TILED-LOWERED-NOT: @memrefCopy

func.func @q15_fir_filter_value(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %init: tensor<?xi16>,
    %index: index) -> i16 {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  %value = tensor.extract %result[%index] : tensor<?xi16>
  return %value : i16
}

func.func @q31_fir_filter_value(
    %input: tensor<?xi32>, %coeffs: tensor<?xi32>, %init: tensor<?xi32>,
    %index: index) -> i32 {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  %value = tensor.extract %result[%index] : tensor<?xi32>
  return %value : i32
}

func.func @q15_fir_filter_shared_coeff_init(
    %input: tensor<?xi16>, %coeffs_and_init: tensor<?xi16>,
    %index: index) -> i16 {
  %result = ondrix.fir_filter %input, %coeffs_and_init, %coeffs_and_init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>) -> tensor<?xi16>
  %value = tensor.extract %result[%index] : tensor<?xi16>
  return %value : i16
}

func.func @f32_fir_filter_value(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %init: tensor<?xf32>,
    %index: index) -> f32 {
  %result = ondrix.fir_filter %input, %coeffs, %init {
    boundary = #ondrix.fir_boundary<valid>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %value = tensor.extract %result[%index] : tensor<?xf32>
  return %value : f32
}

func.func @q15_proven_fir_filter_value(
    %input: tensor<?xi16>, %init: tensor<?xi16>, %index: index) -> i16 {
  %coeffs = arith.constant dense<[-4096, 2048, -1024, 512,
                                   -256, 128, -64, 32]> : tensor<8xi16>
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<8xi16>, tensor<?xi16>) -> tensor<?xi16>
  %value = tensor.extract %result[%index] : tensor<?xi16>
  return %value : i16
}

func.func @q31_proven_fir_filter_value(
    %input: tensor<?xi32>, %init: tensor<?xi32>, %index: index) -> i32 {
  %coeffs = arith.constant dense<[67108864, -33554432, 16777216, -8388608]>
      : tensor<4xi32>
  %result = ondrix.fir_filter %input, %coeffs, %init {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    boundary = #ondrix.fir_boundary<valid>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>, tensor<4xi32>, tensor<?xi32>) -> tensor<?xi32>
  %value = tensor.extract %result[%index] : tensor<?xi32>
  return %value : i32
}
