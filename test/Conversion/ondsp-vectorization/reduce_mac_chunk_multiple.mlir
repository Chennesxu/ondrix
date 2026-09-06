// RUN: ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4 chunk-multiple=4" | FileCheck %s
// RUN: ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4 chunk-multiple=4 pair-fold-squares=false" | FileCheck %s --check-prefix=UNFOLDED

// A chunk wider than the extent would leave the vector loop empty and move the
// whole reduction into the scalar tail, so each extent gets the widest chunk it
// can actually fill.
func.func @fills_the_widest_chunk(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<64xi16>, %rhs: memref<64xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<64xi16>, memref<64xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @fills_the_widest_chunk
// CHECK: vector.load {{.*}} : memref<64xi16>, vector<16xi16>

func.func @steps_down_to_the_extent(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<9xi16>, %rhs: memref<9xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<9xi16>, memref<9xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// Nine elements admit width 8 and no more; a width-16 chunk here would vectorize
// nothing at all.
// CHECK-LABEL: func.func @steps_down_to_the_extent
// CHECK: vector.load {{.*}} : memref<9xi16>, vector<8xi16>
// CHECK: scf.for
// CHECK: ondsp.mac

// A dynamic extent cannot be shown to fill a wide chunk, so it keeps one
// machine vector rather than risking an empty vector loop.
func.func @dynamic_keeps_one_vector(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<?xi16>, %rhs: memref<?xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<?xi16>, memref<?xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: func.func @dynamic_keeps_one_vector
// CHECK: vector.load {{.*}} : memref<?xi16>, vector<4xi16>

// A wrapping sum of squares folds term pairs in i32 and widens them unsigned:
// two nonnegative squares reach at most 2^31, which the unsigned reading holds.
// CHECK-LABEL: func.func @wrapping_sum_of_squares
// CHECK: scf.for {{.*}} iter_args(%[[LANES:.*]] = %{{.*}}) -> (vector<4xi64>)
// CHECK: %[[X:.*]] = vector.load %{{.*}} : memref<64xi16>, vector<16xi16>
// CHECK: vector.extract_strided_slice %[[X]] {offsets = [0], sizes = [8], strides = [1]} : vector<16xi16> to vector<8xi16>
// CHECK: %[[SLICE:.*]] = arith.muli {{.*}} : vector<8xi32>
// CHECK: %[[EVEN:.*]] = vector.shuffle %[[SLICE]], %[[SLICE]] [0, 2, 4, 6]
// CHECK: %[[ODD:.*]] = vector.shuffle %[[SLICE]], %[[SLICE]] [1, 3, 5, 7]
// CHECK: %[[PAIR:.*]] = arith.addi %[[EVEN]], %[[ODD]] : vector<4xi32>
// CHECK: %[[WIDE:.*]] = arith.extui %[[PAIR]] : vector<4xi32> to vector<4xi64>
// CHECK: arith.addi %[[LANES]], %[[WIDE]] : vector<4xi64>
// CHECK: vector.extract_strided_slice {{.*}} {offsets = [8], sizes = [8], strides = [1]}
// CHECK-NOT: arith.extsi {{.*}} to vector<4xi64>
// CHECK: vector.reduction <add>
// Declared off, each product widens on its own and no even/odd fold is formed.
// UNFOLDED-LABEL: func.func @wrapping_sum_of_squares
// UNFOLDED: %[[PRODUCTS:.*]] = arith.muli {{.*}} : vector<16xi32>
// UNFOLDED-NOT: vector.shuffle
// UNFOLDED: %[[SLICE:.*]] = vector.extract_strided_slice %[[PRODUCTS]] {offsets = [0], sizes = [4], strides = [1]} : vector<16xi32> to vector<4xi32>
// UNFOLDED: arith.extsi %[[SLICE]] : vector<4xi32> to vector<4xi64>
// UNFOLDED-NOT: vector.shuffle
// UNFOLDED: vector.reduction <add>
func.func @wrapping_sum_of_squares(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>,
    %input: memref<64xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap> {
  %result = ondsp.reduce_mac %initial, %input, %input {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>, memref<64xi16>, memref<64xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
}
