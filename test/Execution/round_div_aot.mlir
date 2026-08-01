// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --implicit-check-not=ondsp.
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/round_div_aot.c %t.o -o %t
// RUN: %t

// Operation-level differential gate for ondsp.round_div. The i16 profiles
// are swept EXHAUSTIVELY (all 65536 inputs per divisor x mode config)
// against an independent __int128 reference: an odd divisor has no
// reachable rounding tie, an even one reaches all four sign x parity tie
// cells inside the sweep, and the power-of-two config must agree bit-exactly
// with ondsp.round_shift of the same policy. The i32/i64 profiles pin the
// saturating rails, the wrap narrowing, and a declared pre-scale whose
// carrier exceeds 64 bits.
// CHECK-NOT: ondrix.

func.func @rd16_d3_floor(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rd16_d3_zero(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<toward_zero>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rd16_d3_tp(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rd16_d3_ne(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 3 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rd16_d6_floor(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 6 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<toward_negative>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rd16_d6_zero(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 6 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<toward_zero>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rd16_d6_tp(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 6 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rd16_d6_ne(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 6 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

// Identity divisor: every mode is the identity on the full domain.
func.func @rd16_d1_ne(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 1 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

// The largest divisor the i16 carrier admits.
func.func @rd16_dmax_ne(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 32767 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

// The power-of-two pair: round_div by 4 and round_shift by 2 spell the same
// declared boundary and must agree bit-exactly over the whole domain.
func.func @rd16_d4_ne(%x: i16) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 4 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i16) -> i16
  return %y : i16
}

func.func @rs16_s2_ne(%x: i16) -> i16 {
  %y = ondsp.round_shift %x {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 2,
                         rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i16) -> i16
  return %y : i16
}

// Saturating narrowing: quotients far outside i16 must clamp to the rails.
func.func @rd32_d5_ne_sat16(%x: i32) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 5 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i32) -> i16
  return %y : i16
}

// Wrap narrowing of the same profile: two's-complement truncation.
func.func @rd32_d5_tp_wrap16(%x: i32) -> i16 {
  %y = ondsp.round_div %x {
    divisor = 5 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_ties_positive>, overflow = #ondsp.overflow<wrap>
  } : (i32) -> i16
  return %y : i16
}

// Even divisor at i32 for the directed sign x parity tie quadrant.
func.func @rd32_d6_ne(%x: i32) -> i32 {
  %y = ondsp.round_div %x {
    divisor = 6 : i64, pre_shift_left = 0 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i32) -> i32
  return %y : i32
}

// Declared pre-scale into an i112 carrier: the exact left shift happens
// AFTER widening, and the whole division runs past 64 bits.
func.func @rd64_prescaled(%x: i64) -> i32 {
  %y = ondsp.round_div %x {
    divisor = 999999937 : i64, pre_shift_left = 48 : i64,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (i64) -> i32
  return %y : i32
}
