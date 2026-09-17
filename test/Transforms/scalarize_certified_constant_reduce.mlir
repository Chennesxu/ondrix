// RUN: ondrix-opt %s --scalarize-ondsp-certified-constant-reduce --split-input-file | FileCheck %s
// RUN: ondrix-opt %s --scalarize-ondsp-certified-constant-reduce=max-elements=1024 --split-input-file | FileCheck %s --check-prefix=WIDE
// RUN: ondrix-opt %s --scalarize-ondsp-certified-constant-reduce=max-unrolled-terms=10 --split-input-file | FileCheck %s --check-prefix=OVER
// RUN: ondrix-opt %s --scalarize-ondsp-certified-constant-reduce="max-elements=1024 scalar-register-bits=64" --split-input-file | FileCheck %s --check-prefix=WIDE64

!sat = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
memref.global "private" constant @pairs : memref<8xi16> = dense<[32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768]>

// Full-scale coefficients certify pairs, not quads: each pair sum stays below
// 2^31 while two pairs may not, so the chain is four 32-bit pair groups.
// CHECK-LABEL: func.func @dot_pairs(
// CHECK: %[[Z:.*]] = ondsp.acc_zero : <storage = i64, frac = 30, signed, update_overflow = wrap>
// CHECK-NOT: update_overflow = saturate
// CHECK: %[[P0:.*]] = arith.muli %{{.*}}, %c32767_i32 : i32
// CHECK: %[[P1:.*]] = arith.muli %{{.*}}, %c-32768_i32 : i32
// CHECK: %[[S0:.*]] = arith.addi %[[P0]], %[[P1]] : i32
// CHECK: ondsp.acc_add_term %[[Z]], %[[S0]] {term_numeric = #ondsp.fixed<signed, storage = i32, frac = 30>}
// CHECK-COUNT-3: ondsp.acc_add_term
// CHECK-NOT: ondsp.acc_add_term
// CHECK-NOT: ondsp.mac
// CHECK: ondsp.acc_export %{{.*}} : (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>) -> i16
// WIDE-LABEL: func.func @dot_pairs(
// WIDE-COUNT-4: ondsp.acc_add_term
// WIDE64-LABEL: func.func @dot_pairs(
// WIDE64: ondsp.reduce_mac
// WIDE64-NOT: ondsp.acc_add_term
// OVER-LABEL: func.func @dot_pairs(
// OVER-NOT: scf.for
// OVER-COUNT-4: ondsp.acc_add_term
func.func @dot_pairs(%x: memref<8xi16>) -> i16 {
  %c = memref.get_global @pairs : memref<8xi16>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out : i16
}

// -----

!sat = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
memref.global "private" constant @small : memref<8xi16> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>

// Small coefficients certify the whole table as one group; a reversed view
// reads the constants back to front, and the group follows the read order.
// CHECK-LABEL: func.func @dot_reversed(
// CHECK: arith.constant 8 : i32
// CHECK: arith.constant 7 : i32
// CHECK-COUNT-7: arith.addi
// CHECK: ondsp.acc_add_term
// CHECK-NOT: ondsp.acc_add_term
// CHECK: ondsp.acc_export
// WIDE-LABEL: func.func @dot_reversed(
// WIDE: ondsp.acc_add_term
// WIDE64-LABEL: func.func @dot_reversed(
// WIDE64: ondsp.reduce_mac
// WIDE64-NOT: ondsp.acc_add_term
// OVER-LABEL: func.func @dot_reversed(
// OVER: ondsp.acc_add_term
func.func @dot_reversed(%x: memref<8xi16>) -> i16 {
  %c = memref.get_global @small : memref<8xi16>
  %r = memref.subview %c[7] [8] [-1] : memref<8xi16> to memref<8xi16, strided<[-1], offset: 7>>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %r {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16, strided<[-1], offset: 7>>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out : i16
}

// -----

!sat = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
memref.global "private" constant @long : memref<70xi16> = dense<3>

// Past the straight-line length the loop steps over groups of sixteen loaded
// coefficients, and the six products it leaves follow singly.
// CHECK-LABEL: func.func @dot_long(
// CHECK: scf.for %{{.*}} = %c0 to %c64 step %c16 iter_args(%[[ACC:.*]] = %{{.*}}) -> (!ondsp.acc<storage = i64, frac = 30, signed, update_overflow = wrap>)
// CHECK: memref.load %{{.*}} : memref<70xi16>
// CHECK: ondsp.acc_add_term %[[ACC]]
// CHECK: scf.yield
// CHECK-COUNT-6: ondsp.acc_add_term
// CHECK-NOT: ondsp.acc_add_term
// CHECK: ondsp.acc_export
// WIDE-LABEL: func.func @dot_long(
// WIDE: scf.for
// A loop-form reduction still has per-term overhead for the block to
// amortize, so the wider machine keeps it and sums the group in the carrier.
// WIDE64-LABEL: func.func @dot_long(
// WIDE64: scf.for %{{.*}} step %c16
// WIDE64: ondsp.acc_add_term %{{.*}} {term_numeric = #ondsp.fixed<signed, storage = i64, frac = 30>}
// WIDE64-NOT: storage = i32
// OVER-LABEL: func.func @dot_long(
// OVER: scf.for
func.func @dot_long(%x: memref<70xi16>) -> i16 {
  %c = memref.get_global @long : memref<70xi16>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<70xi16>, memref<70xi16>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out : i16
}

// -----

!sat = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
!wrap = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
memref.global "private" constant @rail : memref<600xi16> = dense<-32768>

// Six hundred full-scale products reach the i40 rail, so the saturating
// reduction keeps its declared expansion even when the table is admitted;
// the wrapping one is exact-modulo and takes single terms in a loop.
// CHECK-LABEL: func.func @rail_reachable(
// CHECK: ondsp.reduce_mac
// CHECK-LABEL: func.func @wrap_rail(
// CHECK: ondsp.reduce_mac
// WIDE-LABEL: func.func @rail_reachable(
// WIDE: ondsp.reduce_mac
// WIDE-LABEL: func.func @wrap_rail(
// WIDE: scf.for %{{.*}} = %c0 to %c600 step %c1
// WIDE: ondsp.acc_add_term
// WIDE-NOT: ondsp.reduce_mac
// WIDE64-LABEL: func.func @rail_reachable(
// WIDE64: ondsp.reduce_mac
// WIDE64-LABEL: func.func @wrap_rail(
// WIDE64: ondsp.acc_add_term
// OVER-LABEL: func.func @rail_reachable(
// OVER: ondsp.reduce_mac
func.func @rail_reachable(%x: memref<600xi16>) -> i16 {
  %c = memref.get_global @rail : memref<600xi16>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<600xi16>, memref<600xi16>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out : i16
}
func.func @wrap_rail(%x: memref<600xi16>) -> i16 {
  %c = memref.get_global @rail : memref<600xi16>
  %z = ondsp.acc_zero : !wrap
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!wrap, memref<600xi16>, memref<600xi16>) -> !wrap
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!wrap) -> i16
  return %out : i16
}

// -----

!sat = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
memref.global "private" constant @small : memref<8xi16> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>

// Runtime coefficients have no certificate, and a result read by anything
// but an export keeps the declared accumulator; both stay for the sibling.
// CHECK-LABEL: func.func @keeps_runtime_and_chained(
// CHECK-COUNT-2: ondsp.reduce_mac
// CHECK: ondsp.mac
// WIDE-LABEL: func.func @keeps_runtime_and_chained(
// WIDE-COUNT-2: ondsp.reduce_mac
// WIDE64-LABEL: func.func @keeps_runtime_and_chained(
// WIDE64-COUNT-2: ondsp.reduce_mac
// OVER-LABEL: func.func @keeps_runtime_and_chained(
// OVER-COUNT-2: ondsp.reduce_mac
func.func @keeps_runtime_and_chained(%x: memref<8xi16>, %runtime: memref<8xi16>, %a: i16, %b: i16) -> (i16, i16) {
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %runtime {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  %c = memref.get_global @small : memref<8xi16>
  %chained = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16>) -> !sat
  %more = ondsp.mac %chained, %a, %b {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, i16, i16) -> !sat
  %out2 = ondsp.acc_export %more {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out, %out2 : i16, i16
}

// -----

!sat = !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
memref.global "private" constant @small : memref<8xi16> = dense<[1, 2, 3, 4, 5, 6, 7, 8]>

// The per-function budget is the sibling's: sixteen terms over a budget of
// ten put both reductions in loop form, each stepping over its one group.
// CHECK-LABEL: func.func @two_reductions(
// CHECK-NOT: scf.for
// CHECK-COUNT-2: ondsp.acc_add_term
// WIDE-LABEL: func.func @two_reductions(
// WIDE-NOT: scf.for
// WIDE64-LABEL: func.func @two_reductions(
// WIDE64: ondsp.reduce_mac
// WIDE64-NOT: ondsp.acc_add_term
// OVER-LABEL: func.func @two_reductions(
// OVER: scf.for %{{.*}} = %c0{{.*}} to %c8{{.*}} step %c8{{.*}} iter_args
// OVER: scf.for %{{.*}} = %c0{{.*}} to %c8{{.*}} step %c8{{.*}} iter_args
func.func @two_reductions(%x: memref<8xi16>, %y: memref<8xi16>) -> (i16, i16) {
  %c = memref.get_global @small : memref<8xi16>
  %z = ondsp.acc_zero : !sat
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16>) -> !sat
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  %acc2 = ondsp.reduce_mac %z, %y, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!sat, memref<8xi16>, memref<8xi16>) -> !sat
  %out2 = ondsp.acc_export %acc2 {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!sat) -> i16
  return %out, %out2 : i16, i16
}
