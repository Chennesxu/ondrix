// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ortumcore_cx_packed_aot.c %t.o -o %t.bin
// RUN: %t.bin

func.func @mul_ntp_wrap_s1(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

func.func @mul_floor_wrap_s1(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

func.func @mul_ntp_wrap_s0(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

func.func @mul_ntp_sat_s18(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 18 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow saturate>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

func.func @mul_floor_wrap_s15(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 15 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

func.func @mul_floor_sat_s15(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 15 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow saturate>,
      layout = #ortumcore<cx_layout imag_hi>} : (i32, i32) -> i32
  return %p : i32
}

func.func @mul_swapped_floor_wrap_s0(%v: i32, %w: i32) -> i32 {
  %p = ortumcore.cx_mul_conj %v, %w {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout real_hi>} : (i32, i32) -> i32
  return %p : i32
}

// Butterfly entry points return the pair packed as (out1 << 32) | out0.
func.func private @pack_pair(%o0: i32, %o1: i32) -> i64 {
  %e0 = arith.extui %o0 : i32 to i64
  %e1 = arith.extui %o1 : i32 to i64
  %c32 = arith.constant 32 : i64
  %hi = arith.shli %e1, %c32 : i64
  %r = arith.ori %hi, %e0 : i64
  return %r : i64
}

func.func @bfly_plain_floor_wrap_s1(%a: i32, %b: i32) -> i64 {
  %o0, %o1 = ortumcore.cx_bfly %a, %b {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      variant = #ortumcore<cx_bfly_variant plain>} : (i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%o0, %o1) : (i32, i32) -> i64
  return %r : i64
}

func.func @bfly_plain_ntp_wrap_s1(%a: i32, %b: i32) -> i64 {
  %o0, %o1 = ortumcore.cx_bfly %a, %b {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow wrap>,
      variant = #ortumcore<cx_bfly_variant plain>} : (i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%o0, %o1) : (i32, i32) -> i64
  return %r : i64
}

func.func @bfly_plain_ntp_sat_s1(%a: i32, %b: i32) -> i64 {
  %o0, %o1 = ortumcore.cx_bfly %a, %b {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow saturate>,
      variant = #ortumcore<cx_bfly_variant plain>} : (i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%o0, %o1) : (i32, i32) -> i64
  return %r : i64
}

func.func @bfly_plain_floor_sat_s0(%a: i32, %b: i32) -> i64 {
  %o0, %o1 = ortumcore.cx_bfly %a, %b {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow saturate>,
      variant = #ortumcore<cx_bfly_variant plain>} : (i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%o0, %o1) : (i32, i32) -> i64
  return %r : i64
}

func.func @bfly_cross_floor_wrap_s1(%a: i32, %b: i32) -> i64 {
  %o0, %o1 = ortumcore.cx_bfly %a, %b {shift = 1 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      variant = #ortumcore<cx_bfly_variant cross>} : (i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%o0, %o1) : (i32, i32) -> i64
  return %r : i64
}

func.func @bfly_cross_ntp_sat_s0(%a: i32, %b: i32) -> i64 {
  %o0, %o1 = ortumcore.cx_bfly %a, %b {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding nearest_ties_positive>,
      overflow = #ortumcore<cx_overflow saturate>,
      variant = #ortumcore<cx_bfly_variant cross>} : (i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%o0, %o1) : (i32, i32) -> i64
  return %r : i64
}

// The -+j composition: a real_hi product fed into the cross butterfly.
func.func @compose_pm_j(%x: i32, %c: i32, %w: i32) -> i64 {
  %p = ortumcore.cx_mul_conj %c, %w {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      layout = #ortumcore<cx_layout real_hi>} : (i32, i32) -> i32
  %o0, %o1 = ortumcore.cx_bfly %x, %p {shift = 0 : i64,
      rounding = #ortumcore<cx_rounding toward_negative>,
      overflow = #ortumcore<cx_overflow wrap>,
      variant = #ortumcore<cx_bfly_variant cross>} : (i32, i32) -> (i32, i32)
  %r = func.call @pack_pair(%o0, %o1) : (i32, i32) -> i64
  return %r : i64
}
