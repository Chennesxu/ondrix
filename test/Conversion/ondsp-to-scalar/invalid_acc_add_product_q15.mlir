// RUN: not ondrix-opt %s --convert-ondsp-q15-to-scalar 2>&1 | FileCheck %s

func.func @unsupported_product(
    %acc: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %product: i64) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  // CHECK: error: 'ondsp.acc_add_product' op Q15 scalar lowering requires signed i32 frac=30 product and signed i40 frac=30 accumulator
  %result = ondsp.acc_add_product %acc, %product {
    product_numeric = #ondsp.fixed<signed, storage = i64, frac = 30>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i64) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}
