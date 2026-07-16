// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
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

func.func @q15_mul_nearest_even_wrap(%lhs: i16, %rhs: i16) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @q15_mul_toward_negative(%lhs: i16, %rhs: i16) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_mul_toward_zero(%lhs: i16, %rhs: i16) -> i16 {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_zero>,
    overflow = #ondsp.overflow<saturate>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_import_mac_nearest_even(%seed: i16, %lhs: i16, %rhs: i16) -> i16 {
  %initial = ondsp.acc_import %seed {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.mac %initial, %lhs, %rhs {
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

func.func @q15_repeat_mac_saturate(%lhs: i16, %rhs: i16, %count: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac %current, %lhs, %rhs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_repeat_mac_wrap(%lhs: i16, %rhs: i16, %count: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    %next = ondsp.mac %current, %lhs, %rhs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @q15_repeat_mac_sub_saturate(%lhs: i16, %rhs: i16, %count: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac_sub %current, %lhs, %rhs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_repeat_mac_sub_wrap(%lhs: i16, %rhs: i16, %count: index) -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  %acc = scf.for %i = %c0 to %count step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) {
    %next = ondsp.mac_sub %current, %lhs, %rhs {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  }
  %result = ondsp.acc_export %acc {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>) -> i16
  return %result : i16
}

func.func @q15_positive_saturation_endpoint() -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %minimum = arith.constant -32768 : i16
  %adjust_lhs = arith.constant -32767 : i16
  %adjust_rhs = arith.constant 1 : i16
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %saturated = scf.for %i = %c0 to %c512 step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac %current, %minimum, %minimum {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %adjusted = ondsp.mac %saturated, %adjust_lhs, %adjust_rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %adjusted {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}

func.func @q15_negative_saturation_endpoint() -> i16 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %minimum = arith.constant -32768 : i16
  %adjust_lhs = arith.constant 32767 : i16
  %adjust_rhs = arith.constant 1 : i16
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %saturated = scf.for %i = %c0 to %c512 step %c1
      iter_args(%current = %zero)
      -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) {
    %next = ondsp.mac_sub %current, %minimum, %minimum {
      numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
      product = #ondsp.product<full>
    } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
    scf.yield %next : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  }
  %adjusted = ondsp.mac %saturated, %adjust_lhs, %adjust_rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %result = ondsp.acc_export %adjusted {
    dst = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<toward_negative>,
    overflow = #ondsp.overflow<wrap>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %result : i16
}
