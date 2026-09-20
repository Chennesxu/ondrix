// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/cx_dot_q31_aot.c %t.o -o %t
// RUN: %t

// The Q31 complex profile has no target route, so this object IS the
// contract: components are i32, the term carrier is two components plus one
// bit, and the i64 accumulator saturates on a corpus that reaches its rail.

!acc = !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>

func.func @cx_dot_q31(%lhs: memref<8xi64>, %rhs: memref<8xi64>) -> i64
    attributes {llvm.emit_c_interface} {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>
  } : (memref<8xi64>, memref<8xi64>) -> (!acc, !acc)
  %a = ondsp.acc_export %re {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (!acc) -> i32
  %b = ondsp.acc_export %im {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (!acc) -> i32
  %wa = arith.extui %a : i32 to i64
  %wb = arith.extui %b : i32 to i64
  %c32 = arith.constant 32 : i64
  %hi = arith.shli %wb, %c32 : i64
  %packed = arith.ori %wa, %hi : i64
  return %packed : i64
}

func.func @cx_corr_q31(%lhs: memref<8xi64>, %rhs: memref<8xi64>) -> i64
    attributes {llvm.emit_c_interface} {
  %re, %im = ondrix.cx_dot %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    layout = #ondsp.cx_layout<packed_i32_imag_hi_real_lo>,
    conjugate
  } : (memref<8xi64>, memref<8xi64>) -> (!acc, !acc)
  %a = ondsp.acc_export %re {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (!acc) -> i32
  %b = ondsp.acc_export %im {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>
  } : (!acc) -> i32
  %wa = arith.extui %a : i32 to i64
  %wb = arith.extui %b : i32 to i64
  %c32 = arith.constant 32 : i64
  %hi = arith.shli %wb, %c32 : i64
  %packed = arith.ori %wa, %hi : i64
  return %packed : i64
}
