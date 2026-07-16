// RUN: ondrix-opt %s --vectorize-ondsp-q15-memref-reduce="vector-width=4" --normalize-ondsp-q15-vector-reduce --convert-ondsp-q15-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | FileCheck %s

func.func @reduce_as1(
    %initial: !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>,
    %lhs: memref<8xi16, 1>, %rhs: memref<8xi16, 1>)
    -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate> {
  %result = ondsp.reduce_mac %initial, %lhs, %rhs {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<8xi16, 1>, memref<8xi16, 1>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
}

// CHECK-LABEL: llvm.func @reduce_as1(
// CHECK-SAME: !llvm.ptr<1>
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> vector<4xi16>
// CHECK-NOT: ondsp.
