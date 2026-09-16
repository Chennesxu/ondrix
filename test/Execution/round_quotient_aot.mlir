// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --implicit-check-not=ondsp.
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/round_quotient_aot.c %t.o -o %t
// RUN: %t

// Operation-level differential gate for ondsp.round_quotient against an
// independent __int128 reference of the contract: every i16 dividend for a
// set of divisors including the non-positive ones, every i16 divisor for a
// set of dividends, the unscaled profile where the even divisor reaches all
// four tie cells, the ratio profile whose ties are provably unreachable, and
// the trapping policy caught in a child process.
// CHECK-NOT: ondrix.

func.func @rq16_ratio_tp(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64, rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>, nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

func.func @rq16_ratio_ne(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64, rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>, nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

func.func @rq16_ratio_floor(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64, rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>, nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

func.func @rq16_ratio_zero(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64, rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>, nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

// Unscaled: the plain rounded integer quotient, where an even divisor
// reaches every tie cell.
func.func @rq16_plain_tp(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 0 : i64, rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>, nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

func.func @rq16_plain_ne(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 0 : i64, rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>, nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i16, i16) -> i16
  return %0 : i16
}

// Wrap narrowing of the ratio into i16 from a wider dividend.
func.func @rq32_ratio_wrap(%x: i32, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64, rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>, nonpositive = #ondsp.nonpositive_divisor<saturate>
  } : (i32, i16) -> i16
  return %0 : i16
}

// The trapping policy: identical to the saturating one on a positive
// divisor, and a stopped process on a divisor that is not.
func.func @rq16_ratio_trap(%x: i16, %y: i16) -> i16 {
  %0 = ondsp.round_quotient %x, %y {
    pre_shift_left = 15 : i64, rounding = #ondsp.rounding<nearest_ties_positive>,
    overflow = #ondsp.overflow<saturate>, nonpositive = #ondsp.nonpositive_divisor<trap>
  } : (i16, i16) -> i16
  return %0 : i16
}
