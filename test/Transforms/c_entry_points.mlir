// RUN: ondrix-opt %s --declare-ondrix-c-entry-points --split-input-file | FileCheck %s

// Two dynamic windows one reduce_mac pairs share an extent group; the names
// come from the arguments' locations.
// CHECK-LABEL: func.func @dot(
// CHECK-SAME: attributes {llvm.emit_c_interface, ondrix.c_entry = {groups = array<i64: 0, 0>, names = ["lhs", "rhs"], signature = (memref<?xf32>, memref<?xf32>) -> f32}}
func.func @dot(%a: memref<?xf32> loc("lhs"), %b: memref<?xf32> loc("rhs")) -> f32
    attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0.0 : f32
  %sum = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fp<format = f32, contract = fma>}
      : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %sum : f32
}

// -----

// Windows no reduction pairs each take their own length under fallback names;
// a static window takes none.
// CHECK-LABEL: func.func @separate(
// CHECK-SAME: ondrix.c_entry = {groups = array<i64: 0, -1, 1>, names = ["a0", "a1", "a2"], signature = (memref<?xi16>, memref<5xi16>, memref<?xi16>) -> i16}
func.func @separate(%a: memref<?xi16>, %b: memref<5xi16>, %c: memref<?xi16>) -> i16
    attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : i16
  return %zero : i16
}

// -----

// A tensor result, a rank-2 buffer, a strided view and an unexported function
// get no record.
// CHECK-LABEL: func.func @full(
// CHECK-NOT: ondrix.c_entry
// CHECK-LABEL: func.func @matrix(
// CHECK-NOT: ondrix.c_entry
// CHECK-LABEL: func.func @strided(
// CHECK-NOT: ondrix.c_entry
// CHECK-LABEL: func.func @internal(
// CHECK-NOT: ondrix.c_entry
func.func @full(%a: memref<?xi16>) -> memref<?xi16> attributes {llvm.emit_c_interface} {
  return %a : memref<?xi16>
}
func.func @matrix(%a: memref<4x16xi16>) -> i16 attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : i16
  return %zero : i16
}
func.func @strided(%a: memref<?xi16, strided<[2]>>) -> i16 attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : i16
  return %zero : i16
}
func.func @internal(%a: memref<?xi16>) -> i16 {
  %zero = arith.constant 0 : i16
  return %zero : i16
}
