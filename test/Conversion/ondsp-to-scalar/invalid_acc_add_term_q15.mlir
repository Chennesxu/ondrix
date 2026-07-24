// RUN: not ondrix-opt %s --convert-ondsp-fixed-to-scalar 2>&1 | FileCheck %s

func.func @unsupported_accumulator(
    %acc: !ondsp.acc<storage = i48, frac = 29, signed, update_overflow = saturate>,
    %product: i64) -> !ondsp.acc<storage = i48, frac = 29, signed, update_overflow = saturate> {
  // CHECK: error: failed to legalize operation 'func.func'
  %result = ondsp.acc_add_term %acc, %product {
    term_numeric = #ondsp.fixed<signed, storage = i64, frac = 29>
  } : (!ondsp.acc<storage = i48, frac = 29, signed, update_overflow = saturate>, i64) -> !ondsp.acc<storage = i48, frac = 29, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i48, frac = 29, signed, update_overflow = saturate>
}
