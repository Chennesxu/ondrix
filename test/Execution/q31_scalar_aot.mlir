// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/q31_scalar_aot.c %t.o -o %t
// RUN: %t

func.func @q31_full_nearest_even(%lhs: i32, %rhs: i32) -> i32 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %acc = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>) -> i32
  return %result : i32
}

func.func @q31_repeat_full_saturate(%lhs: i32, %rhs: i32, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>) {
    %next = ondsp.mac %current, %lhs, %rhs {
      numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = saturate>) -> i32
  return %result : i32
}

func.func @q31_repeat_full_wrap(%lhs: i32, %rhs: i32, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>) {
    %next = ondsp.mac %current, %lhs, %rhs {
      numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>, i32, i32) -> !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
    scf.yield %next : !ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 31>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i64, frac = 62, signed, update_overflow = wrap>) -> i32
  return %result : i32
}

func.func @q31_high_raw_q30(%lhs: i32, %rhs: i32) -> i32 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 30>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %result : i32
}

func.func @q31_high_raw_sub_q30(%lhs: i32, %rhs: i32) -> i32 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.mac_sub %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<high_raw>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i32, i32) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i32, frac = 30>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i32
  return %result : i32
}
