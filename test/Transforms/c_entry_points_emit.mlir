// RUN: ondrix-opt %s --emit-ondrix-c-entry-points --split-input-file | FileCheck %s
// RUN: ondrix-opt %s --emit-ondrix-c-entry-points=checked=true --split-input-file | FileCheck %s --check-prefix=CHECKED

// One length after both pointers of the declared group, bounded so the i16
// buffer fits the signed range in bytes; each descriptor is the pointer
// twice, offset zero, the length and stride one; the record moves to the
// wrapper and leaves the descriptor wrapper's copy.
// CHECK-LABEL: llvm.func @dot(
// CHECK-NOT: ondrix.
// CHECK-LABEL: llvm.func @_mlir_ciface_dot(
// CHECK-NOT: ondrix.
// CHECK-LABEL: llvm.func @ondrix_dot(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr, %[[B:.*]]: !llvm.ptr, %[[N:.*]]: i64) -> i16
// CHECK-SAME: ondrix.c_entry = {groups = array<i64: 0, 0>, names = ["lhs", "rhs"], signature = (memref<?xi16>, memref<?xi16>) -> i16}
// CHECK: %[[MAX:.*]] = llvm.mlir.constant(4611686018427387903 : i64)
// CHECK: %[[OK:.*]] = llvm.icmp "ule" %[[N]], %[[MAX]]
// CHECK: llvm.cond_br %[[OK]], ^[[CALL:.*]], ^[[REFUSE:.*]]
// CHECK: ^[[CALL]]:
// CHECK-NOT: llvm.ptrtoint
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[R:.*]] = llvm.call @dot(%[[A]], %[[A]], %[[ZERO]], %[[N]], %[[ONE]], %[[B]], %[[B]], %[[ZERO]], %[[N]], %[[ONE]])
// CHECK: llvm.return %[[R]]
// CHECK: ^[[REFUSE]]:
// CHECK: llvm.call @puts
// CHECK: llvm.call @abort
// CHECK: llvm.unreachable
// Checked: after the length bound, a buffer with elements must not be null
// and the two half-open byte ranges must not intersect, each with its own
// message; the call is reached only past all three.
// CHECKED-DAG: llvm.mlir.global private constant @{{.*}}("ondrix_dot: a length exceeds the addressable range\00")
// CHECKED-DAG: llvm.mlir.global private constant @{{.*}}("ondrix_dot: a null buffer has elements\00")
// CHECKED-DAG: llvm.mlir.global private constant @{{.*}}("ondrix_dot: two buffers overlap\00")
// CHECKED-LABEL: llvm.func @ondrix_dot(
// CHECKED-SAME: %[[A:.*]]: !llvm.ptr, %[[B:.*]]: !llvm.ptr, %[[N:.*]]: i64) -> i16
// CHECKED: llvm.cond_br %{{.*}}, ^[[RANGE_OK:.*]], ^{{.*}}
// CHECKED: ^[[RANGE_OK]]:
// CHECKED: %[[PA:.*]] = llvm.ptrtoint %[[A]]
// CHECKED: %[[TWO:.*]] = llvm.mlir.constant(2 : i64)
// CHECKED: %[[BYTES_A:.*]] = llvm.mul %[[N]], %[[TWO]]
// CHECKED: %[[END_A:.*]] = llvm.add %[[PA]], %[[BYTES_A]]
// CHECKED: %[[PB:.*]] = llvm.ptrtoint %[[B]]
// CHECKED: %[[END_B:.*]] = llvm.add %[[PB]], %{{.*}}
// CHECKED: %[[START:.*]] = llvm.intr.umax(%[[PA]], %[[PB]])
// CHECKED: %[[STOP:.*]] = llvm.intr.umin(%[[END_A]], %[[END_B]])
// CHECKED: %[[DISJOINT:.*]] = llvm.icmp "uge" %[[START]], %[[STOP]]
// CHECKED: llvm.cond_br %{{.*}}, ^[[NULL_OK:.*]], ^{{.*}}
// CHECKED: ^[[NULL_OK]]:
// CHECKED-NEXT: llvm.cond_br %[[DISJOINT]], ^[[CALL:.*]], ^{{.*}}
// CHECKED: ^[[CALL]]:
// CHECKED: llvm.call @dot(%[[A]], %[[A]], %{{.*}}, %[[N]], %{{.*}}, %[[B]], %[[B]], %{{.*}}, %[[N]], %{{.*}})
llvm.func @abort()
llvm.func @puts(!llvm.ptr)
llvm.func @dot(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i64, %a3: i64, %a4: i64,
               %b0: !llvm.ptr, %b1: !llvm.ptr, %b2: i64, %b3: i64, %b4: i64) -> i16
    attributes {llvm.emit_c_interface, ondrix.c_entry = {groups = array<i64: 0, 0>, names = ["lhs", "rhs"],
                signature = (memref<?xi16>, memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}
llvm.func @_mlir_ciface_dot(%a: !llvm.ptr, %b: !llvm.ptr) -> i16
    attributes {llvm.emit_c_interface, ondrix.c_entry = {groups = array<i64: 0, 0>, names = ["lhs", "rhs"],
                signature = (memref<?xi16>, memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}

// -----

// A static extent needs neither a length nor a range check, and the index
// width is whatever the expanded kernel carries; separate extent groups each
// take a length right after their own pointer.
// CHECK-LABEL: llvm.func @ondrix_window(
// CHECK-SAME: %[[W:.*]]: !llvm.ptr) -> i32
// CHECK-NOT: llvm.cond_br
// CHECK: %[[FIVE:.*]] = llvm.mlir.constant(5 : i32)
// CHECK: llvm.call @window(%[[W]], %[[W]], %{{.*}}, %[[FIVE]], %{{.*}})
// CHECK-LABEL: llvm.func @ondrix_filter(
// CHECK-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr, %{{.*}}: i32) -> i16
llvm.func @window(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i32, %a3: i32, %a4: i32) -> i32
    attributes {ondrix.c_entry = {groups = array<i64: -1>, names = ["w"], signature = (memref<5xi32>) -> i32}} {
  %zero = llvm.mlir.constant(0 : i32) : i32
  llvm.return %zero : i32
}
llvm.func @filter(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i32, %a3: i32, %a4: i32,
                  %b0: !llvm.ptr, %b1: !llvm.ptr, %b2: i32, %b3: i32, %b4: i32) -> i16
    attributes {ondrix.c_entry = {groups = array<i64: 0, 1>, names = ["input", "coefficients"],
                signature = (memref<?xi16>, memref<?xi16>) -> i16}} {
  %zero = llvm.mlir.constant(0 : i16) : i16
  llvm.return %zero : i16
}

// -----

// A rank-2 buffer expands to two sizes and two row-major strides, a result
// out-parameter is one more pointer, and the entry returns nothing.
// CHECK-LABEL: llvm.func @ondrix_matrix(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr, %[[OUT:.*]]: !llvm.ptr) attributes
// CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
// CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
// CHECK-DAG: %[[THREE:.*]] = llvm.mlir.constant(3 : i64)
// CHECK-DAG: %[[FOUR:.*]] = llvm.mlir.constant(4 : i64)
// CHECK-DAG: %[[EIGHT:.*]] = llvm.mlir.constant(8 : i64)
// CHECK: llvm.call @matrix(%[[A]], %[[A]], %[[ZERO]], %[[FOUR]], %[[EIGHT]], %[[EIGHT]], %[[ONE]], %[[OUT]], %[[OUT]], %[[ZERO]], %[[THREE]], %[[ONE]])
// CHECK-NEXT: llvm.return
llvm.func @matrix(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i64, %a3: i64, %a4: i64, %a5: i64, %a6: i64,
                  %o0: !llvm.ptr, %o1: !llvm.ptr, %o2: i64, %o3: i64, %o4: i64)
    attributes {ondrix.c_entry = {groups = array<i64: -1, -1>, names = ["a", "output"],
                signature = (memref<4x8xi16>) -> memref<3xi16>}} {
  llvm.return
}
