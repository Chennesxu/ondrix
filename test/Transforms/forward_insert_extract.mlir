// RUN: ondrix-opt %s --forward-ondrix-insert-extract | FileCheck %s

// Value-identity forwarding: a constant-index read is replaced by the scalar
// of the nearest enclosing constant-index write to the same indices. Nothing
// the walk cannot prove is rewritten, and the unforwarded program is always
// the safe state.

// The nearest matching write wins: a later write to the same index shadows
// the earlier one, so the read observes the last stored scalar.
// CHECK-LABEL: func.func @nearest_write_wins(
// CHECK-SAME:      %{{.*}}: tensor<4xi32>, %{{.*}}: i32, %{{.*}}: i32, %[[LAST:.*]]: i32
// CHECK-NOT:     tensor.extract
// CHECK:         return %[[LAST]]
func.func @nearest_write_wins(%base: tensor<4xi32>, %first: i32, %other: i32, %last: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %write0 = tensor.insert %first into %base[%c0] : tensor<4xi32>
  %write1 = tensor.insert %other into %write0[%c1] : tensor<4xi32>
  %write2 = tensor.insert %last into %write1[%c0] : tensor<4xi32>
  %read = tensor.extract %write2[%c0] : tensor<4xi32>
  return %read : i32
}

// Every index must agree, not just the leading one.
// CHECK-LABEL: func.func @multi_dim_requires_all_indices(
// CHECK-SAME:      %{{.*}}: tensor<2x2xi32>, %[[VALUE:.*]]: i32
// CHECK:         %[[ROW:.*]] = tensor.extract
// CHECK:         return %[[VALUE]], %[[ROW]]
func.func @multi_dim_requires_all_indices(%base: tensor<2x2xi32>, %value: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %write = tensor.insert %value into %base[%c0, %c1] : tensor<2x2xi32>
  %matched = tensor.extract %write[%c0, %c1] : tensor<2x2xi32>
  %other = tensor.extract %write[%c0, %c0] : tensor<2x2xi32>
  return %matched, %other : i32, i32
}

// A dynamic read index cannot be compared with any write index.
// CHECK-LABEL: func.func @dynamic_read_index_untouched
// CHECK:         tensor.insert
// CHECK:         tensor.extract
func.func @dynamic_read_index_untouched(%base: tensor<4xi32>, %value: i32, %index: index) -> i32 {
  %c1 = arith.constant 1 : index
  %write = tensor.insert %value into %base[%c1] : tensor<4xi32>
  %read = tensor.extract %write[%index] : tensor<4xi32>
  return %read : i32
}

// A dynamic write may target the read index, so the walk stops at it instead
// of looking past it at the matching constant write below.
// CHECK-LABEL: func.func @dynamic_write_blocks_walk
// CHECK:         tensor.insert
// CHECK:         tensor.insert
// CHECK:         tensor.extract
func.func @dynamic_write_blocks_walk(%base: tensor<4xi32>, %first: i32, %second: i32,
                                     %index: index) -> i32 {
  %c0 = arith.constant 0 : index
  %write0 = tensor.insert %first into %base[%c0] : tensor<4xi32>
  %write1 = tensor.insert %second into %write0[%index] : tensor<4xi32>
  %read = tensor.extract %write1[%c0] : tensor<4xi32>
  return %read : i32
}

// A slice write between the read and a matching scalar write is not a
// tensor.insert, so the walk stops at it. That stop is the safety property,
// not a missed opportunity: this slice covers the read index, and looking
// past it to the matching write below would forward a shadowed value.
// CHECK-LABEL: func.func @insert_slice_blocks_walk
// CHECK:         tensor.insert
// CHECK:         tensor.insert_slice
// CHECK:         tensor.extract
func.func @insert_slice_blocks_walk(%base: tensor<4xi32>, %value: i32,
                                    %slice: tensor<2xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %write = tensor.insert %value into %base[%c0] : tensor<4xi32>
  %overlaid = tensor.insert_slice %slice into %write[0] [2] [1]
      : tensor<2xi32> into tensor<4xi32>
  %read = tensor.extract %overlaid[%c0] : tensor<4xi32>
  return %read : i32
}

// The walk reaches a function argument without a matching write; the read
// observes incoming data and is left alone.
// CHECK-LABEL: func.func @unmatched_read_reaches_argument
// CHECK:         tensor.insert
// CHECK:         tensor.extract
func.func @unmatched_read_reaches_argument(%base: tensor<4xi32>, %value: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %write = tensor.insert %value into %base[%c1] : tensor<4xi32>
  %read = tensor.extract %write[%c0] : tensor<4xi32>
  return %read : i32
}

// The walk reaches `tensor.empty` without a matching write. Reading
// uninitialized storage is an upstream defect, so the pass invents no value
// and the read survives.
// CHECK-LABEL: func.func @unmatched_read_reaches_empty
// CHECK:         tensor.empty
// CHECK:         tensor.insert
// CHECK:         tensor.extract
func.func @unmatched_read_reaches_empty(%value: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %base = tensor.empty() : tensor<4xi32>
  %write = tensor.insert %value into %base[%c1] : tensor<4xi32>
  %read = tensor.extract %write[%c0] : tensor<4xi32>
  return %read : i32
}

// Mixed chain: the two written indices forward and the unwritten one does
// not, so the intermediate tensor legitimately stays materialized.
// CHECK-LABEL: func.func @mixed_chain_partial_forward(
// CHECK-SAME:      %{{.*}}: tensor<4xi32>, %[[FIRST:.*]]: i32, %[[SECOND:.*]]: i32
// CHECK:         %[[UNWRITTEN:.*]] = tensor.extract
// CHECK-NOT:     tensor.extract
// CHECK:         return %[[FIRST]], %[[SECOND]], %[[UNWRITTEN]]
func.func @mixed_chain_partial_forward(%base: tensor<4xi32>, %first: i32,
                                       %second: i32) -> (i32, i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %write0 = tensor.insert %first into %base[%c0] : tensor<4xi32>
  %write1 = tensor.insert %second into %write0[%c1] : tensor<4xi32>
  %read0 = tensor.extract %write1[%c0] : tensor<4xi32>
  %read1 = tensor.extract %write1[%c1] : tensor<4xi32>
  %read2 = tensor.extract %write1[%c2] : tensor<4xi32>
  return %read0, %read1, %read2 : i32, i32, i32
}
