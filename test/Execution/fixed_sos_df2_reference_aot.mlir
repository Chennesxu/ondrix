// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/fixed_sos_df2_reference_aot.c %t.o -o %t
// RUN: %t

// This test-only ABI returns output, next d1, or next d2 for part 0, 1, or 2.
// It composes the candidate DF-II equation from existing Ondsp numeric ops;
// there is intentionally no public fixed recursive-filter operation yet.

// CHECK-NOT: ondsp.

func.func @fixed_sos_df2_q15_sat_update_mixed_export(
    %input: i32, %scale: i32, %b0: i32, %b1: i32, %b2: i32,
    %a1: i32, %a2: i32, %d1: i32, %d2: i32, %part: i32) -> i32 {
  %input_i16 = arith.trunci %input : i32 to i16
  %scale_i16 = arith.trunci %scale : i32 to i16
  %b0_i16 = arith.trunci %b0 : i32 to i16
  %b1_i16 = arith.trunci %b1 : i32 to i16
  %b2_i16 = arith.trunci %b2 : i32 to i16
  %a1_i16 = arith.trunci %a1 : i32 to i16
  %a2_i16 = arith.trunci %a2 : i32 to i16
  %d1_i16 = arith.trunci %d1 : i32 to i16
  %d2_i16 = arith.trunci %d2 : i32 to i16
  %state0 = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %state1 = ondsp.mac %state0, %input_i16, %scale_i16 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       i16, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %state2 = ondsp.mac %state1, %d1_i16, %a1_i16 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       i16, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %state3 = ondsp.mac %state2, %d2_i16, %a2_i16 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       i16, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %next_d1 = ondsp.acc_export %state3 {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
      -> i16

  %output0 = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %output1 = ondsp.mac %output0, %next_d1, %b0_i16 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       i16, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %output2 = ondsp.mac %output1, %d1_i16, %b1_i16 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       i16, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %output3 = ondsp.mac %output2, %d2_i16, %b2_i16 {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
       i16, i16)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %output = ondsp.acc_export %output3 {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
      -> i16

  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %is_output = arith.cmpi eq, %part, %c0 : i32
  %is_d1 = arith.cmpi eq, %part, %c1 : i32
  %state = arith.select %is_d1, %next_d1, %d1_i16 : i16
  %result_i16 = arith.select %is_output, %output, %state : i16
  %result = arith.extsi %result_i16 : i16 to i32
  return %result : i32
}

func.func @fixed_sos_df2_q31_saturate(
    %input: i32, %scale: i32, %b0: i32, %b1: i32, %b2: i32,
    %a1: i32, %a2: i32, %d1: i32, %d2: i32, %part: i32) -> i32 {
  %state0 = ondsp.acc_zero
      : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %state1 = ondsp.mac %state0, %input, %scale {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %state2 = ondsp.mac %state1, %d1, %a1 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %state3 = ondsp.mac %state2, %d2, %a2 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %next_d1 = ondsp.acc_export %state3 {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>)
      -> i32

  %output0 = ondsp.acc_zero
      : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %output1 = ondsp.mac %output0, %next_d1, %b0 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %output2 = ondsp.mac %output1, %d1, %b1 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %output3 = ondsp.mac %output2, %d2, %b2 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %output = ondsp.acc_export %output3 {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>)
      -> i32

  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %is_output = arith.cmpi eq, %part, %c0 : i32
  %is_d1 = arith.cmpi eq, %part, %c1 : i32
  %state = arith.select %is_d1, %next_d1, %d1 : i32
  %result = arith.select %is_output, %output, %state : i32
  return %result : i32
}

func.func @fixed_sos_df2_q31_wrap(
    %input: i32, %scale: i32, %b0: i32, %b1: i32, %b2: i32,
    %a1: i32, %a2: i32, %d1: i32, %d2: i32, %part: i32) -> i32 {
  %state0 = ondsp.acc_zero
      : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %state1 = ondsp.mac %state0, %input, %scale {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %state2 = ondsp.mac %state1, %d1, %a1 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %state3 = ondsp.mac %state2, %d2, %a2 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %next_d1 = ondsp.acc_export %state3 {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>) -> i32

  %output0 = ondsp.acc_zero
      : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %output1 = ondsp.mac %output0, %next_d1, %b0 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %output2 = ondsp.mac %output1, %d1, %b1 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %output3 = ondsp.mac %output2, %d2, %b2 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %output = ondsp.acc_export %output3 {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>) -> i32

  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %is_output = arith.cmpi eq, %part, %c0 : i32
  %is_d1 = arith.cmpi eq, %part, %c1 : i32
  %state = arith.select %is_d1, %next_d1, %d1 : i32
  %result = arith.select %is_output, %output, %state : i32
  return %result : i32
}

// Exposes the test-only raw state accumulator so the C harness can distinguish
// saturation after each update from saturation deferred to the final sum.
func.func @fixed_sos_df2_q31_sat_update_raw(
    %input: i32, %scale: i32, %d1: i32, %a1: i32, %d2: i32, %a2: i32) -> i64 {
  %state0 = ondsp.acc_zero
      : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %state1 = ondsp.mac %state0, %input, %scale {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %state2 = ondsp.mac %state1, %d1, %a1 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %state3 = ondsp.mac %state2, %d2, %a2 {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>,
       i32, i32)
      -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %raw = builtin.unrealized_conversion_cast %state3
      : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate> to i64
  return %raw : i64
}
