// RUN: ondrix-opt %s --convert-ondsp-lane-split-to-ortumcore --split-input-file | FileCheck %s

memref.global "private" constant @normalized_taps : memref<8xi16> = dense<[-1200, 2500, 6000, 9100, 9100, 6000, 2500, -1200]>

// A unity-gain table: both lanes and their sum fit one word, so the split
// merges with one add; the ordered chain stays for a misaligned stream.
// CHECK-LABEL: func.func @normalized
// CHECK: memref.extract_aligned_pointer_as_index %{{.*}}
// CHECK: scf.if
// CHECK: memref.assume_alignment %{{.*}}, 4
// CHECK-COUNT-4: ortumcore.dmac
// CHECK: %[[S0:.*]] = ortumcore.acc_out %{{.*}} {shift = 0 : i64}
// CHECK: %[[S1:.*]] = ortumcore.acc_out %{{.*}} {shift = 0 : i64}
// CHECK: arith.addi %[[S0]], %[[S1]] : i32
// CHECK: } else {
// CHECK-COUNT-8: ondsp.mac
func.func @normalized(%x: memref<8xi16>) -> i16 {
  %table = memref.get_global @normalized_taps : memref<8xi16>
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<8xi16>
  %t0 = arith.constant 0 : index
  %c0 = memref.load %table[%t0] : memref<8xi16>
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<8xi16>
  %t1 = arith.constant 1 : index
  %c1 = memref.load %table[%t1] : memref<8xi16>
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<8xi16>
  %t2 = arith.constant 2 : index
  %c2 = memref.load %table[%t2] : memref<8xi16>
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<8xi16>
  %t3 = arith.constant 3 : index
  %c3 = memref.load %table[%t3] : memref<8xi16>
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<8xi16>
  %t4 = arith.constant 4 : index
  %c4 = memref.load %table[%t4] : memref<8xi16>
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i5 = arith.constant 5 : index
  %x5 = memref.load %x[%i5] : memref<8xi16>
  %t5 = arith.constant 5 : index
  %c5 = memref.load %table[%t5] : memref<8xi16>
  %acc6 = ondsp.mac %acc5, %x5, %c5 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i6 = arith.constant 6 : index
  %x6 = memref.load %x[%i6] : memref<8xi16>
  %t6 = arith.constant 6 : index
  %c6 = memref.load %table[%t6] : memref<8xi16>
  %acc7 = ondsp.mac %acc6, %x6, %c6 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i7 = arith.constant 7 : index
  %x7 = memref.load %x[%i7] : memref<8xi16>
  %t7 = arith.constant 7 : index
  %c7 = memref.load %table[%t7] : memref<8xi16>
  %acc8 = ondsp.mac %acc7, %x7, %c7 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc8 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// Five terms: the last pairs with an exact zero product on lane 1.
// CHECK-LABEL: func.func @odd_tail
// CHECK: %[[Z:.*]] = arith.constant 0 : i16
// CHECK-COUNT-2: ortumcore.dmac
// CHECK: ortumcore.dmac {{.*}}, %[[Z]], %[[Z]] :
func.func @odd_tail(%x: memref<6xi16>) -> i16 {
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<6xi16>
  %c0 = arith.constant 3000 : i16
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<6xi16>
  %c1 = arith.constant -2000 : i16
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<6xi16>
  %c2 = arith.constant 5000 : i16
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<6xi16>
  %c3 = arith.constant 7000 : i16
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<6xi16>
  %c4 = arith.constant 1000 : i16
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc5 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// Lane 0 carries 32767 + 32767 + 2: its lower bound is exactly -2^31.
// CHECK-LABEL: func.func @lane_at_word_bound
// CHECK: ortumcore.dmac
func.func @lane_at_word_bound(%x: memref<6xi16>) -> i16 {
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<6xi16>
  %c0 = arith.constant 32767 : i16
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<6xi16>
  %c1 = arith.constant 1 : i16
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<6xi16>
  %c2 = arith.constant 32767 : i16
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<6xi16>
  %c3 = arith.constant 1 : i16
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<6xi16>
  %c4 = arith.constant 2 : i16
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i5 = arith.constant 5 : index
  %x5 = memref.load %x[%i5] : memref<6xi16>
  %c5 = arith.constant 1 : i16
  %acc6 = ondsp.mac %acc5, %x5, %c5 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// One unit more on lane 0 leaves the word, so the chain stays ordered.
// CHECK-LABEL: func.func @lane_past_word_bound
// CHECK-NOT: ortumcore.dmac
// CHECK-NOT: scf.if
// CHECK: return
func.func @lane_past_word_bound(%x: memref<6xi16>) -> i16 {
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<6xi16>
  %c0 = arith.constant 32767 : i16
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<6xi16>
  %c1 = arith.constant 1 : i16
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<6xi16>
  %c2 = arith.constant 32767 : i16
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<6xi16>
  %c3 = arith.constant 1 : i16
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<6xi16>
  %c4 = arith.constant 3 : i16
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i5 = arith.constant 5 : index
  %x5 = memref.load %x[%i5] : memref<6xi16>
  %c5 = arith.constant 1 : i16
  %acc6 = ondsp.mac %acc5, %x5, %c5 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc6 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// Each lane fits a word but the sum does not, and a shift-0 readout has no
// halving to absorb it.
// CHECK-LABEL: func.func @readout_zero_needs_the_sum
// CHECK-NOT: ortumcore.dmac
// CHECK: return
func.func @readout_zero_needs_the_sum(%x: memref<8xi16>) -> i32 {
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<8xi16>
  %c0 = arith.constant 16384 : i16
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<8xi16>
  %c1 = arith.constant 16384 : i16
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<8xi16>
  %c2 = arith.constant 16384 : i16
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<8xi16>
  %c3 = arith.constant 16384 : i16
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<8xi16>
  %c4 = arith.constant 16384 : i16
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i5 = arith.constant 5 : index
  %x5 = memref.load %x[%i5] : memref<8xi16>
  %c5 = arith.constant 16384 : i16
  %acc6 = ondsp.mac %acc5, %x5, %c5 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i6 = arith.constant 6 : index
  %x6 = memref.load %x[%i6] : memref<8xi16>
  %c6 = arith.constant 16384 : i16
  %acc7 = ondsp.mac %acc6, %x6, %c6 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i7 = arith.constant 7 : index
  %x7 = memref.load %x[%i7] : memref<8xi16>
  %c7 = arith.constant 16384 : i16
  %acc8 = ondsp.mac %acc7, %x7, %c7 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc8 {dst = #ondsp.fixed<signed, storage = i32, frac = 30>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<toward_negative>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i32
  return %r : i32
}

// -----

// The same lanes under a shift-15 export merge exactly through the halves.
// CHECK-LABEL: func.func @readout_fifteen_merges_by_halves
// CHECK: %[[S0:.*]] = ortumcore.acc_out %{{.*}} {shift = 0 : i64}
// CHECK: %[[S1:.*]] = ortumcore.acc_out %{{.*}} {shift = 0 : i64}
// CHECK: arith.shrsi %[[S0]]
// CHECK: arith.shrsi %[[S1]]
// CHECK: arith.andi %[[S0]], %[[S1]]
func.func @readout_fifteen_merges_by_halves(%x: memref<8xi16>) -> i16 {
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<8xi16>
  %c0 = arith.constant 16384 : i16
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<8xi16>
  %c1 = arith.constant 16384 : i16
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<8xi16>
  %c2 = arith.constant 16384 : i16
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<8xi16>
  %c3 = arith.constant 16384 : i16
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i4 = arith.constant 4 : index
  %x4 = memref.load %x[%i4] : memref<8xi16>
  %c4 = arith.constant 16384 : i16
  %acc5 = ondsp.mac %acc4, %x4, %c4 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i5 = arith.constant 5 : index
  %x5 = memref.load %x[%i5] : memref<8xi16>
  %c5 = arith.constant 16384 : i16
  %acc6 = ondsp.mac %acc5, %x5, %c5 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i6 = arith.constant 6 : index
  %x6 = memref.load %x[%i6] : memref<8xi16>
  %c6 = arith.constant 16384 : i16
  %acc7 = ondsp.mac %acc6, %x6, %c6 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i7 = arith.constant 7 : index
  %x7 = memref.load %x[%i7] : memref<8xi16>
  %c7 = arith.constant 16384 : i16
  %acc8 = ondsp.mac %acc7, %x7, %c7 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc8 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// Two runtime operands give the certificate nothing to read.
// CHECK-LABEL: func.func @runtime_coefficients
// CHECK-NOT: ortumcore.dmac
// CHECK: return
func.func @runtime_coefficients(%x: memref<4xi16>, %y: memref<4xi16>) -> i16 {
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<4xi16>
  %c0 = memref.load %y[%i0] : memref<4xi16>
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<4xi16>
  %c1 = memref.load %y[%i1] : memref<4xi16>
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<4xi16>
  %c2 = memref.load %y[%i2] : memref<4xi16>
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<4xi16>
  %c3 = memref.load %y[%i3] : memref<4xi16>
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc4 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// Starting at element 1 no pair is one word of the stream.
// CHECK-LABEL: func.func @odd_start
// CHECK-NOT: ortumcore.dmac
// CHECK: return
func.func @odd_start(%x: memref<6xi16>) -> i16 {
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 1 : index
  %x0 = memref.load %x[%i0] : memref<6xi16>
  %c0 = arith.constant 3000 : i16
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 2 : index
  %x1 = memref.load %x[%i1] : memref<6xi16>
  %c1 = arith.constant -2000 : i16
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 3 : index
  %x2 = memref.load %x[%i2] : memref<6xi16>
  %c2 = arith.constant 5000 : i16
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 4 : index
  %x3 = memref.load %x[%i3] : memref<6xi16>
  %c3 = arith.constant 7000 : i16
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc4 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// Runtime taps under a declared bound of 2^16: every sum is at most
// 2^15 * (2^16 - 1), inside a word, and both pointers are tested.
// CHECK-LABEL: func.func @declared_bound_splits
// CHECK: %[[P0:.*]] = memref.extract_aligned_pointer_as_index %arg0
// CHECK: %[[P1:.*]] = memref.extract_aligned_pointer_as_index %[[T:.*]] :
// CHECK: arith.ori %[[P0]], %[[P1]]
// CHECK: memref.assume_alignment %arg0, 4
// CHECK: memref.assume_alignment %[[T]], 4
// CHECK-COUNT-2: ortumcore.dmac
func.func @declared_bound_splits(%x: memref<4xi16>, %taps: memref<4xi16>) -> i16 {
  %bounded = ondsp.assume_l1_bound %taps {bound = 65536 : i64} : memref<4xi16>
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<4xi16>
  %c0 = memref.load %bounded[%i0] : memref<4xi16>
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<4xi16>
  %c1 = memref.load %bounded[%i1] : memref<4xi16>
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<4xi16>
  %c2 = memref.load %bounded[%i2] : memref<4xi16>
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<4xi16>
  %c3 = memref.load %bounded[%i3] : memref<4xi16>
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc4 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}

// -----

// One raw unit more and the bound admits 2^15 * 2^16 = 2^31, one past a word.
// CHECK-LABEL: func.func @declared_bound_past_a_word
// CHECK-NOT: ortumcore.dmac
// CHECK: return
func.func @declared_bound_past_a_word(%x: memref<4xi16>, %taps: memref<4xi16>) -> i16 {
  %bounded = ondsp.assume_l1_bound %taps {bound = 65537 : i64} : memref<4xi16>
  %acc0 = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i0 = arith.constant 0 : index
  %x0 = memref.load %x[%i0] : memref<4xi16>
  %c0 = memref.load %bounded[%i0] : memref<4xi16>
  %acc1 = ondsp.mac %acc0, %x0, %c0 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i1 = arith.constant 1 : index
  %x1 = memref.load %x[%i1] : memref<4xi16>
  %c1 = memref.load %bounded[%i1] : memref<4xi16>
  %acc2 = ondsp.mac %acc1, %x1, %c1 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i2 = arith.constant 2 : index
  %x2 = memref.load %x[%i2] : memref<4xi16>
  %c2 = memref.load %bounded[%i2] : memref<4xi16>
  %acc3 = ondsp.mac %acc2, %x2, %c2 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %i3 = arith.constant 3 : index
  %x3 = memref.load %x[%i3] : memref<4xi16>
  %c3 = memref.load %bounded[%i3] : memref<4xi16>
  %acc4 = ondsp.mac %acc3, %x3, %c3 {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, i16, i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %r = ondsp.acc_export %acc4 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %r : i16
}
