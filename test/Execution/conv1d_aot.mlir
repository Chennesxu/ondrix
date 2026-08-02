// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/conv1d_aot.c %t.o -lm -o %t
// RUN: %t
// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --lower-ondsp-f32-reduce-to-scalar --convert-ondsp-fixed-to-scalar --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.bufferized.mlir
// RUN: ondrix-translate %t.bufferized.mlir --mlir-to-llvmir > %t.bufferized.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.bufferized.ll -o %t.bufferized.o
// RUN: cc -ffp-contract=off %S/Inputs/conv1d_aot.c %t.bufferized.o -lm -o %t.bufferized
// RUN: %t.bufferized
// RUN: ondrix-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=2" --normalize-ondsp-fixed-vector-reduce --lower-ondsp-f32-reduce-to-scalar --convert-ondsp-fixed-to-scalar --convert-vector-to-llvm --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.vector.mlir
// RUN: ondrix-translate %t.vector.mlir --mlir-to-llvmir > %t.vector.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.vector.ll -o %t.vector.o
// RUN: cc -ffp-contract=off %S/Inputs/conv1d_aot.c %t.vector.o -lm -o %t.vector
// RUN: %t.vector

// CHECK-NOT: ondrix.
// CHECK-NOT: ondsp.

func.func @q15_convolution_value(
    %x0: i16, %x1: i16, %x2: i16, %x3: i16, %x4: i16, %x5: i16,
    %k0: i16, %k1: i16, %k2: i16, %index: index) -> i16 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5
      : tensor<6xi16>
  %kernel = tensor.from_elements %k0, %k1, %k2 : tensor<3xi16>
  %init = tensor.empty() : tensor<4xi16>
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<6xi16>, tensor<3xi16>, tensor<4xi16>) -> tensor<4xi16>
  %value = tensor.extract %result[%index] : tensor<4xi16>
  return %value : i16
}

func.func @q15_correlation_value(
    %x0: i16, %x1: i16, %x2: i16, %x3: i16, %x4: i16, %x5: i16,
    %k0: i16, %k1: i16, %k2: i16, %index: index) -> i16 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5
      : tensor<6xi16>
  %kernel = tensor.from_elements %k0, %k1, %k2 : tensor<3xi16>
  %init = tensor.empty() : tensor<4xi16>
  %result = ondrix.conv1d %input, %kernel, %init {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<6xi16>, tensor<3xi16>, tensor<4xi16>) -> tensor<4xi16>
  %value = tensor.extract %result[%index] : tensor<4xi16>
  return %value : i16
}

func.func @f32_convolution_value(
    %x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %x5: f32,
    %k0: f32, %k1: f32, %k2: f32, %index: index) -> f32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5
      : tensor<6xf32>
  %kernel = tensor.from_elements %k0, %k1, %k2 : tensor<3xf32>
  %init = tensor.empty() : tensor<4xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<6xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<4xf32>
  %value = tensor.extract %result[%index] : tensor<4xf32>
  return %value : f32
}

func.func @f32_correlation_value(
    %x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %x5: f32,
    %k0: f32, %k1: f32, %k2: f32, %index: index) -> f32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5
      : tensor<6xf32>
  %kernel = tensor.from_elements %k0, %k1, %k2 : tensor<3xf32>
  %init = tensor.empty() : tensor<4xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<6xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<4xf32>
  %value = tensor.extract %result[%index] : tensor<4xf32>
  return %value : f32
}

// The off contract rounds every tap product before the accumulator observes
// it, so these twins must not be lowered through a fused update. They pair
// with the fused functions above to gate both f32 contract modes.
func.func @f32_convolution_value_off(
    %x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %x5: f32,
    %k0: f32, %k1: f32, %k2: f32, %index: index) -> f32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5
      : tensor<6xf32>
  %kernel = tensor.from_elements %k0, %k1, %k2 : tensor<3xf32>
  %init = tensor.empty() : tensor<4xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<convolution>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<6xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<4xf32>
  %value = tensor.extract %result[%index] : tensor<4xf32>
  return %value : f32
}

func.func @f32_correlation_value_off(
    %x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %x5: f32,
    %k0: f32, %k1: f32, %k2: f32, %index: index) -> f32 {
  %input = tensor.from_elements %x0, %x1, %x2, %x3, %x4, %x5
      : tensor<6xf32>
  %kernel = tensor.from_elements %k0, %k1, %k2 : tensor<3xf32>
  %init = tensor.empty() : tensor<4xf32>
  %result = ondrix.conv1d %input, %kernel, %init {
    mode = #ondrix.conv1d_mode<correlation>,
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<6xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<4xf32>
  %value = tensor.extract %result[%index] : tensor<4xf32>
  return %value : f32
}
