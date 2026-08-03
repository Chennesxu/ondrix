// RUN: ondrix-opt %s --lower-ondsp-f32-reduce-to-scalar | FileCheck %s

// The f32 scalar lowering leaves every other reduction for its own consumer.
// f64 used to be the case here; it is now refused at the verifier instead, so
// what remains to preserve is the fixed profile.

func.func @preserve_q15(%lhs: memref<8xi16>, %rhs: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %zero = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %0 = ondsp.reduce_mac %zero, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %0 : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @preserve_q15
// CHECK: ondsp.reduce_mac
