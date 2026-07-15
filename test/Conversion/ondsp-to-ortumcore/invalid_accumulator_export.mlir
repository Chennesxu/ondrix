// RUN: not ondrix-opt %s --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

func.func @acc_export(%acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16 {
  // CHECK: policy-bearing accumulator export is unsupported by ortumcore lowering until target export semantics are proven equivalent
  %0 = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %0 : i16
}
