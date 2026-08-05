// RUN: ondrix-opt %s --convert-ondrix-to-ondsp > %t.ondsp.mlir
// RUN: FileCheck %s --input-file=%t.ondsp.mlir --check-prefix=LOWERED
// RUN: ondrix-opt %t.ondsp.mlir --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=FINAL
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/sos_filter_df2_fixed_aot.c %t.o -o %t
// RUN: %t
// RUN: cc %S/Inputs/sos_filter_df2_fixed_mismatch.c %t.o -o %t.mismatch
// RUN: not --crash %t.mismatch zero
// RUN: not --crash %t.mismatch scales
// RUN: not --crash %t.mismatch state

// LOWERED-LABEL: func.func @sos_fixed_q15_output_value
// LOWERED-COUNT-6: ondsp.mac
// LOWERED-COUNT-2: ondsp.acc_export
// LOWERED-NOT: ondrix.sos_filter_df2_fixed
// LOWERED-LABEL: func.func @sos_fixed_q31_output_value
// LOWERED-COUNT-6: ondsp.mac
// LOWERED-COUNT-2: ondsp.acc_export
// LOWERED-NOT: ondrix.sos_filter_df2_fixed
// LOWERED-LABEL: func.func @sos_fixed_q15_wrap_output_value
// LOWERED-COUNT-2: ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = wrap>
// LOWERED: ondsp.acc_export
// LOWERED-SAME: overflow = #ondsp.overflow<saturate>
// LOWERED-SAME: rounding = #ondsp.rounding<nearest_even>
// LOWERED-LABEL: func.func @sos_fixed_q31_saturate_output_value
// LOWERED-COUNT-2: ondsp.acc_zero : <storage = i64, frac = 62, signed, update_overflow = saturate>
// LOWERED: ondsp.acc_export
// LOWERED-SAME: overflow = #ondsp.overflow<wrap>
// LOWERED-SAME: rounding = #ondsp.rounding<toward_zero>
// FINAL-NOT: ondrix.
// FINAL-NOT: ondsp.

func.func @sos_fixed_q15_output_value(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>, %index: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<wrap>, output_rounding = #ondsp.rounding<toward_zero>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  %value = tensor.extract %output[%index] : tensor<?xi16>
  %extended = arith.extsi %value : i16 to i32
  return %extended : i32
}

func.func @sos_fixed_q15_state_value(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>,
    %section: index, %slot: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<wrap>, output_rounding = #ondsp.rounding<toward_zero>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  %value = tensor.extract %next[%section, %slot] : tensor<?x2xi16>
  %extended = arith.extsi %value : i16 to i32
  return %extended : i32
}

func.func @sos_fixed_q31_output_value(
    %input: tensor<?xi32>, %coeffs: tensor<?x5xi32>,
    %scales: tensor<?xi32>, %state: tensor<?x2xi32>, %index: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<toward_negative>, product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<wrap>, state_rounding = #ondsp.rounding<toward_zero>
  } : (tensor<?xi32>, tensor<?x5xi32>, tensor<?xi32>, tensor<?x2xi32>)
      -> (tensor<?xi32>, tensor<?x2xi32>)
  %value = tensor.extract %output[%index] : tensor<?xi32>
  return %value : i32
}

func.func @sos_fixed_q31_state_value(
    %input: tensor<?xi32>, %coeffs: tensor<?x5xi32>,
    %scales: tensor<?xi32>, %state: tensor<?x2xi32>,
    %section: index, %slot: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<toward_negative>, product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<wrap>, state_rounding = #ondsp.rounding<toward_zero>
  } : (tensor<?xi32>, tensor<?x5xi32>, tensor<?xi32>, tensor<?x2xi32>)
      -> (tensor<?xi32>, tensor<?x2xi32>)
  %value = tensor.extract %next[%section, %slot] : tensor<?x2xi32>
  return %value : i32
}

func.func @sos_fixed_q15_wrap_output_value(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>, %index: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<nearest_even>, product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<wrap>,
    state_rounding = #ondsp.rounding<toward_negative>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  %value = tensor.extract %output[%index] : tensor<?xi16>
  %extended = arith.extsi %value : i16 to i32
  return %extended : i32
}

func.func @sos_fixed_q15_wrap_state_value(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>,
    %section: index, %slot: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<nearest_even>, product = #ondsp.product<full>,
    state_overflow = #ondsp.overflow<wrap>,
    state_rounding = #ondsp.rounding<toward_negative>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  %value = tensor.extract %next[%section, %slot] : tensor<?x2xi16>
  %extended = arith.extsi %value : i16 to i32
  return %extended : i32
}

func.func @sos_fixed_q31_saturate_output_value(
    %input: tensor<?xi32>, %coeffs: tensor<?x5xi32>,
    %scales: tensor<?xi32>, %state: tensor<?x2xi32>, %index: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<wrap>, output_rounding = #ondsp.rounding<toward_zero>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>, tensor<?x5xi32>, tensor<?xi32>, tensor<?x2xi32>)
      -> (tensor<?xi32>, tensor<?x2xi32>)
  %value = tensor.extract %output[%index] : tensor<?xi32>
  return %value : i32
}

func.func @sos_fixed_q31_saturate_state_value(
    %input: tensor<?xi32>, %coeffs: tensor<?x5xi32>,
    %scales: tensor<?xi32>, %state: tensor<?x2xi32>,
    %section: index, %slot: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_overflow = #ondsp.overflow<wrap>, output_rounding = #ondsp.rounding<toward_zero>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_even>
  } : (tensor<?xi32>, tensor<?x5xi32>, tensor<?xi32>, tensor<?x2xi32>)
      -> (tensor<?xi32>, tensor<?x2xi32>)
  %value = tensor.extract %next[%section, %slot] : tensor<?x2xi32>
  return %value : i32
}

func.func @sos_fixed_q15_ties_positive_output_value(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>, %index: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<nearest_ties_positive>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  %value = tensor.extract %output[%index] : tensor<?xi16>
  %extended = arith.extsi %value : i16 to i32
  return %extended : i32
}

func.func @sos_fixed_q15_ties_positive_state_value(
    %input: tensor<?xi16>, %coeffs: tensor<?x5xi16>,
    %scales: tensor<?xi16>, %state: tensor<?x2xi16>,
    %section: index, %slot: index) -> i32 {
  %output, %next = ondrix.sos_filter_df2_fixed %input, %coeffs, %scales, %state {
    accumulator = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    output_overflow = #ondsp.overflow<saturate>,
    output_rounding = #ondsp.rounding<nearest_ties_positive>,
    product = #ondsp.product<full>, state_overflow = #ondsp.overflow<saturate>,
    state_rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<?xi16>, tensor<?x5xi16>, tensor<?xi16>, tensor<?x2xi16>)
      -> (tensor<?xi16>, tensor<?x2xi16>)
  %value = tensor.extract %next[%section, %slot] : tensor<?x2xi16>
  %extended = arith.extsi %value : i16 to i32
  return %extended : i32
}
