// RUN: ondrix-opt %s --vectorize-ondsp-fixed-memref-reduce="vector-width=4 chunk-multiple=4" | FileCheck %s

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
