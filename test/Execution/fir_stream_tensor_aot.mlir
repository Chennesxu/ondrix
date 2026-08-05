// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.mlir
// RUN: FileCheck %s --check-prefix=DEALLOC < %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/fir_stream_tensor_aot.c %t.o -lm -o %t
// RUN: %t
// RUN: cc -ffp-contract=off %S/Inputs/fir_stream_tensor_mismatch.c %t.o -lm -o %t.mismatch
// RUN: not --crash %t.mismatch coefficients
// RUN: not --crash %t.mismatch state
// RUN: not --crash %t.mismatch output
// RUN: not --crash %t.mismatch next
// RUN: ondrix-opt %s --decompose-ondrix-fir-stream --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --cse --canonicalize --vectorize-ondsp-fixed-memref-reduce="vector-width=2" --normalize-ondsp-fixed-vector-reduce --lower-ondsp-f32-reduce-to-scalar --canonicalize > %t.vector.mlir
// RUN: FileCheck %s --check-prefix=STREAM-VECTOR < %t.vector.mlir
// RUN: ondrix-opt %t.vector.mlir --lower-rank-one-memref-copy-to-scf --convert-ondsp-fixed-to-scalar --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.vector.llvm.mlir
// RUN: FileCheck %s --check-prefix=VECTOR-DEALLOC < %t.vector.llvm.mlir
// RUN: ondrix-translate %t.vector.llvm.mlir --mlir-to-llvmir > %t.vector.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.vector.ll -o %t.vector.o
// RUN: cc -ffp-contract=off %S/Inputs/fir_stream_tensor_aot.c %t.vector.o -lm -o %t.vector
// RUN: %t.vector
// RUN: cc -ffp-contract=off %S/Inputs/fir_stream_tensor_mismatch.c %t.vector.o -lm -o %t.vector.mismatch
// RUN: not --crash %t.vector.mismatch coefficients
// RUN: not --crash %t.vector.mismatch state
// RUN: not --crash %t.vector.mismatch output
// RUN: not --crash %t.vector.mismatch next

// LOWERED-NOT: ondrix.
// LOWERED-NOT: ondsp.
// LOWERED-NOT: tensor.

// DEALLOC-LABEL: llvm.func @q15_stream_output_value
// DEALLOC: llvm.call @malloc
// DEALLOC: llvm.call @free

// STREAM-VECTOR-LABEL: func.func @q15_stream_output_value
// STREAM-VECTOR: scf.if
// STREAM-VECTOR: vector.load {{.*}}vector<2xi16>
// STREAM-VECTOR: arith.muli {{.*}} : vector<2xi32>
// STREAM-VECTOR-NOT: ondrix.fir_stream

// STREAM-VECTOR-LABEL: func.func @q31_stream_output_value
// STREAM-VECTOR: scf.if
// STREAM-VECTOR: vector.load {{.*}}vector<2xi32>
// STREAM-VECTOR: arith.muli {{.*}} : vector<2xi64>
// STREAM-VECTOR-NOT: ondrix.fir_stream

// STREAM-VECTOR-LABEL: func.func @f32_stream_output_value
// STREAM-VECTOR: math.fma
// STREAM-VECTOR-NOT: ondrix.fir_stream

// STREAM-VECTOR-LABEL: func.func @f32_stream_off_output_value
// STREAM-VECTOR: arith.mulf
// STREAM-VECTOR: arith.addf
// STREAM-VECTOR-NOT: ondrix.fir_stream

// VECTOR-DEALLOC-LABEL: llvm.func @q15_stream_output_value
// VECTOR-DEALLOC: %[[EXTENDED:.*]] = llvm.call @malloc
// VECTOR-DEALLOC: %[[STATE_SNAPSHOT:.*]] = llvm.call @malloc
// VECTOR-DEALLOC: llvm.call @free(%[[STATE_SNAPSHOT]])
// VECTOR-DEALLOC: %[[INPUT_SNAPSHOT:.*]] = llvm.call @malloc
// VECTOR-DEALLOC: llvm.call @free(%[[INPUT_SNAPSHOT]])
// VECTOR-DEALLOC: %[[OUTPUT:.*]] = llvm.call @malloc
// VECTOR-DEALLOC: llvm.call @free(%[[EXTENDED]])
// VECTOR-DEALLOC: llvm.call @free(%[[OUTPUT]])

// DEALLOC-LABEL: llvm.func @q15_stream_state_value
// DEALLOC: llvm.call @malloc
// DEALLOC: llvm.call @free

func.func @q15_stream_output_value(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %state: tensor<?xi16>,
    %index: index) -> i16 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>)
      -> (tensor<?xi16>, tensor<?xi16>)
  %value = tensor.extract %output[%index] : tensor<?xi16>
  return %value : i16
}

func.func @q15_stream_state_value(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %state: tensor<?xi16>,
    %index: index) -> i16 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>)
      -> (tensor<?xi16>, tensor<?xi16>)
  %value = tensor.extract %next[%index] : tensor<?xi16>
  return %value : i16
}

func.func @q31_stream_output_value(
    %input: tensor<?xi32>, %coeffs: tensor<?xi32>, %state: tensor<?xi32>,
    %index: index) -> i32 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>)
      -> (tensor<?xi32>, tensor<?xi32>)
  %value = tensor.extract %output[%index] : tensor<?xi32>
  return %value : i32
}

func.func @q31_stream_state_value(
    %input: tensor<?xi32>, %coeffs: tensor<?xi32>, %state: tensor<?xi32>,
    %index: index) -> i32 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>)
      -> (tensor<?xi32>, tensor<?xi32>)
  %value = tensor.extract %next[%index] : tensor<?xi32>
  return %value : i32
}

func.func @f32_stream_output_value(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %state: tensor<?xf32>,
    %index: index) -> f32 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>)
  %value = tensor.extract %output[%index] : tensor<?xf32>
  return %value : f32
}

func.func @f32_stream_state_value(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %state: tensor<?xf32>,
    %index: index) -> f32 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>)
  %value = tensor.extract %next[%index] : tensor<?xf32>
  return %value : f32
}

func.func @f32_stream_off_output_value(
    %input: tensor<?xf32>, %coeffs: tensor<?xf32>, %state: tensor<?xf32>,
    %index: index) -> f32 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>)
  %value = tensor.extract %output[%index] : tensor<?xf32>
  return %value : f32
}

func.func @q15_stream_static_output_value(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %state: tensor<?xi16>,
    %index: index) -> i16 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>)
      -> (tensor<4xi16>, tensor<?xi16>)
  %value = tensor.extract %output[%index] : tensor<4xi16>
  return %value : i16
}

func.func @q15_stream_static_state_value(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %state: tensor<?xi16>,
    %index: index) -> i16 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>)
      -> (tensor<?xi16>, tensor<2xi16>)
  %value = tensor.extract %next[%index] : tensor<2xi16>
  return %value : i16
}

func.func @q15_stream_output_value_ties_positive(
    %input: tensor<?xi16>, %coeffs: tensor<?xi16>, %state: tensor<?xi16>,
    %index: index) -> i16 {
  %output, %next = ondrix.fir_stream %input, %coeffs, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>,
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    overflow = #ondsp.overflow<saturate>,
    product = #ondsp.product<full>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<?xi16>, tensor<?xi16>, tensor<?xi16>)
      -> (tensor<?xi16>, tensor<?xi16>)
  %value = tensor.extract %output[%index] : tensor<?xi16>
  return %value : i16
}
