// RUN: ondrix-opt %s --convert-ondsp-q15-to-scalar --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q15_scalar_aot.c %t.o -o %t
// RUN: %t

func.func @q15_mul_nearest_even(%lhs: i16, %rhs: i16) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}
