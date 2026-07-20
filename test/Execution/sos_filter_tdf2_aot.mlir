// RUN: ondrix-opt %s --convert-ondrix-to-ondsp > %t.ondsp.mlir
// RUN: FileCheck %s --input-file=%t.ondsp.mlir --check-prefix=LOWERED
// RUN: ondrix-opt %t.ondsp.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -ffp-contract=off %S/Inputs/sos_filter_tdf2_aot.c %t.o -lm -o %t
// RUN: %t
// RUN: cc %S/Inputs/sos_filter_tdf2_mismatch.c %t.o -lm -o %t.mismatch
// RUN: not --crash %t.mismatch scales
// RUN: not --crash %t.mismatch state

// LOWERED-LABEL: func.func @sos_fma_output_value
// LOWERED: math.fma
// LOWERED-NOT: ondrix.sos_filter_tdf2
// LOWERED-LABEL: func.func @sos_fma_state_value
// LOWERED: math.fma
// LOWERED-NOT: ondrix.sos_filter_tdf2
// LOWERED-LABEL: func.func @sos_off_output_value
// LOWERED: arith.mulf
// LOWERED: arith.addf
// LOWERED-NOT: ondrix.sos_filter_tdf2
// LOWERED-LABEL: func.func @sos_off_state_value
// LOWERED: arith.mulf
// LOWERED: arith.addf
// LOWERED-NOT: ondrix.sos_filter_tdf2

func.func @sos_fma_output_value(
    %input: tensor<?xf32>, %coeffs: tensor<?x5xf32>,
    %scales: tensor<?xf32>, %state: tensor<?x2xf32>, %index: index) -> f32 {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?x5xf32>, tensor<?xf32>, tensor<?x2xf32>)
      -> (tensor<?xf32>, tensor<?x2xf32>)
  %value = tensor.extract %output[%index] : tensor<?xf32>
  return %value : f32
}

func.func @sos_fma_state_value(
    %input: tensor<?xf32>, %coeffs: tensor<?x5xf32>,
    %scales: tensor<?xf32>, %state: tensor<?x2xf32>,
    %section: index, %slot: index) -> f32 {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = fma>
  } : (tensor<?xf32>, tensor<?x5xf32>, tensor<?xf32>, tensor<?x2xf32>)
      -> (tensor<?xf32>, tensor<?x2xf32>)
  %value = tensor.extract %next[%section, %slot] : tensor<?x2xf32>
  return %value : f32
}

func.func @sos_off_output_value(
    %input: tensor<?xf32>, %coeffs: tensor<?x5xf32>,
    %scales: tensor<?xf32>, %state: tensor<?x2xf32>, %index: index) -> f32 {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<?xf32>, tensor<?x5xf32>, tensor<?xf32>, tensor<?x2xf32>)
      -> (tensor<?xf32>, tensor<?x2xf32>)
  %value = tensor.extract %output[%index] : tensor<?xf32>
  return %value : f32
}

func.func @sos_off_state_value(
    %input: tensor<?xf32>, %coeffs: tensor<?x5xf32>,
    %scales: tensor<?xf32>, %state: tensor<?x2xf32>,
    %section: index, %slot: index) -> f32 {
  %output, %next = ondrix.sos_filter_tdf2 %input, %coeffs, %scales, %state {
    numeric = #ondsp.fp<format = f32, contract = off>
  } : (tensor<?xf32>, tensor<?x5xf32>, tensor<?xf32>, tensor<?x2xf32>)
      -> (tensor<?xf32>, tensor<?x2xf32>)
  %value = tensor.extract %next[%section, %slot] : tensor<?x2xf32>
  return %value : f32
}
