// RUN: ondrix-opt %s --vectorize-ondsp-fp-filter-outputs="vector-width=8" | FileCheck %s

// The divisor 0x00000001 is 2^-149: a power of two whose f32 reciprocal
// overflows to +inf, where 0/x is 0 but 0*inf is NaN — the mean keeps its
// division.
// CHECK-LABEL: func.func @subnormal_divisor_keeps_division
// CHECK-NOT: arith.constant 0x7F800000
// CHECK: arith.divf {{.*}} : vector<8xf32>
// CHECK-NOT: arith.constant 0x7F800000
func.func @subnormal_divisor_keeps_division(%input: memref<64xf32>) -> memref<57xf32> {
  %cst = arith.constant 0x00000001 : f32
  %c57 = arith.constant 57 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<57xf32>
  scf.for %n = %c0 to %c57 step %c1 {
    %seed = memref.load %input[%n] : memref<64xf32>
    %sum = scf.for %k = %c1 to %c8 step %c1 iter_args(%acc = %seed) -> (f32) {
      %pos = arith.addi %n, %k : index
      %sample = memref.load %input[%pos] : memref<64xf32>
      %next = arith.addf %acc, %sample : f32
      scf.yield %next : f32
    }
    %mean = arith.divf %sum, %cst : f32
    memref.store %mean, %alloc[%n] : memref<57xf32>
  }
  return %alloc : memref<57xf32>
}
