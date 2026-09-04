// RUN: ondrix-opt %s --convert-ondsp-fixed-to-scalar | FileCheck %s
// RUN: ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4" --convert-ondsp-fixed-to-scalar | FileCheck %s --check-prefix=VEC

// A requantized full product: the exact i64 product is rounded right by the
// declared shift before it joins the wrapping accumulator at frac 62 - 3.
// CHECK-LABEL: func.func @q31_shifted_product_mac(
// CHECK: %[[P:.*]] = arith.muli {{.*}} : i64
// CHECK: %[[C3:.*]] = arith.constant 3 : i64
// CHECK: %[[Q:.*]] = arith.shrsi %[[P]], %[[C3]] : i64
// CHECK: arith.trunci %[[P]] : i64 to i3
// CHECK: arith.cmpi ugt
// CHECK: arith.cmpi eq
// CHECK: %[[TERM:.*]] = arith.addi %[[Q]], %{{.*}} : i64
// CHECK: arith.addi %{{.*}}, %[[TERM]] : i64
func.func @q31_shifted_product_mac(
    %acc: !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>,
    %lhs: i32, %rhs: i32) -> !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap> {
  %r = ondsp.mac %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full, shift = 3, rounding = nearest_even>
  } : (!ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>, i32, i32) -> !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>
  return %r : !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>
}

// The wrap route carries the same requantized terms as i64 lane sums.
// VEC-LABEL: func.func @q31_shifted_product_reduce(
// VEC: arith.muli {{.*}} : vector<4xi64>
// VEC: arith.shrsi {{.*}} : vector<4xi64>
// VEC: arith.trunci {{.*}} : vector<4xi64> to vector<4xi3>
// VEC: vector.reduction <add>
func.func @q31_shifted_product_reduce(
    %acc: !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>,
    %lhs: memref<8xi32>, %rhs: memref<8xi32>) -> !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap> {
  %r = ondsp.reduce_mac %acc, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    product = #ondsp.product<full, shift = 3, rounding = nearest_even>
  } : (!ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>, memref<8xi32>, memref<8xi32>) -> !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>
  return %r : !ondsp.acc<storage = i64, frac = 59, signed, update_overflow = wrap>
}
