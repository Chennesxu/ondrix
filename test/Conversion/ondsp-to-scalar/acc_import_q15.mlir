// RUN: ondrix-opt %s --convert-ondsp-q15-to-scalar | FileCheck %s

func.func @import_q15(%input: i16)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %acc = ondsp.acc_import %input {
    src = #ondsp.fixed<signed, storage = i16, frac = 15>
  } : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %acc : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @import_q15(
// CHECK-SAME: %[[INPUT:.*]]: i16) -> i40
// CHECK: %[[EXTENDED:.*]] = arith.extsi %[[INPUT]] : i16 to i40
// CHECK: %[[SHIFT:.*]] = arith.constant 15 : i40
// CHECK: %[[ACC:.*]] = arith.shli %[[EXTENDED]], %[[SHIFT]] : i40
// CHECK: return %[[ACC]] : i40
// CHECK-NOT: ondsp.
