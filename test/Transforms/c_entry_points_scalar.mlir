// RUN: ondrix-opt %s --declare-ondrix-c-entry-points | FileCheck %s --check-prefix=RECORD
// RUN: ondrix-opt %s --ondrix-default-pipeline | FileCheck %s --check-prefix=LLVM
// RUN: ondrix-opt %s --ondrix-default-pipeline | ondrix-translate --mlir-to-ondrix-c-header | FileCheck %s --check-prefix=HEADER

// Scalars retain their positions, and bool has an explicit byte ABI.
// RECORD-LABEL: func.func @scalars(
// RECORD-SAME: groups = array<i64: -1, -1, -1, -1>, names = ["a", "b", "f", "flag"], signature = (i32, i64, f32, i1) -> i1
// LLVM-LABEL: llvm.func @ondrix_scalars(
// LLVM-SAME: %[[A:.*]]: i32, %[[B:.*]]: i64, %[[F:.*]]: f32, %[[FLAG:.*]]: i8) -> i8
// LLVM: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8)
// LLVM: %[[BOOL:.*]] = llvm.icmp "ne" %[[FLAG]], %[[ZERO]] : i8
// LLVM: %[[R:.*]] = llvm.call @scalars(%[[A]], %[[B]], %[[F]], %[[BOOL]])
// LLVM: %[[BYTE:.*]] = llvm.zext %[[R]] : i1 to i8
// LLVM: llvm.return %[[BYTE]] : i8
// HEADER: uint8_t ondrix_scalars(int32_t a, int64_t b, float f, uint8_t flag);
func.func @scalars(%a: i32 loc("a"), %b: i64 loc("b"), %f: f32 loc("f"), %flag: i1 loc("flag")) -> i1 attributes {llvm.emit_c_interface} {
  return %flag : i1
}

// A reduction's shared length still follows its last buffer, even with
// scalars between the buffers and at either end of the input list.
// RECORD-LABEL: func.func @mixed(
// RECORD-SAME: groups = array<i64: -1, 0, -1, 0, -1>, names = ["seed", "lhs", "flag", "rhs", "extra"], signature = (f32, memref<?xf32>, i1, memref<?xf32>, i64) -> f32
// LLVM-LABEL: llvm.func @ondrix_mixed(
// LLVM-SAME: %{{.*}}: f32, %{{.*}}: !llvm.ptr, %{{.*}}: i8, %{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64) -> f32
// HEADER: float ondrix_mixed(float seed, const float *lhs, uint8_t flag, const float *rhs, uint64_t length, int64_t extra);
func.func @mixed(%seed: f32 loc("seed"), %lhs: memref<?xf32> loc("lhs"), %flag: i1 loc("flag"), %rhs: memref<?xf32> loc("rhs"), %extra: i64 loc("extra")) -> f32 attributes {llvm.emit_c_interface} {
  %sum = ondsp.reduce_mac %seed, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %sum : f32
}

// Scalar inputs do not change trailing output-buffer placement.
// RECORD-LABEL: func.func @fill(
// RECORD-SAME: groups = array<i64: -1, -1>, names = ["value", "output"], signature = (i32) -> memref<2xi32>
// HEADER: void ondrix_fill(int32_t value, int32_t *output);
func.func @fill(%value: i32 loc("value")) -> memref<2xi32> attributes {llvm.emit_c_interface} {
  %buffer = memref.alloc() : memref<2xi32>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  memref.store %value, %buffer[%zero] : memref<2xi32>
  memref.store %value, %buffer[%one] : memref<2xi32>
  return %buffer : memref<2xi32>
}
