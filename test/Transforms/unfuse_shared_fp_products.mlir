// RUN: ondrix-opt %s --unfuse-ondsp-shared-fp-products | FileCheck %s

// A product nothing else reads stays fused: that is the case the permission
// exists for.
func.func @keeps_a_private_product(%a: f32, %b: f32, %c: f32) -> f32 {
  %0 = math.fma %a, %b, %c {ondsp.fast_used = ["fuse_multiply_add"]} : f32
  return %0 : f32
}

// CHECK-LABEL: func.func @keeps_a_private_product
// CHECK: math.fma
// CHECK-NOT: arith.mulf

// The same product is needed on its own, so fusing recomputes it; the ordered
// form costs the same instruction and multiplies once.
func.func @relaxes_a_shared_product(%a: f32, %b: f32, %c: f32) -> (f32, f32) {
  %p = arith.mulf %a, %b : f32
  %0 = math.fma %a, %b, %c {ondsp.fast_used = ["fuse_multiply_add"]} : f32
  return %0, %p : f32, f32
}

// CHECK-LABEL: func.func @relaxes_a_shared_product
// CHECK: %[[P:.*]] = arith.mulf
// CHECK: arith.addf %{{.*}}, %[[P]]
// CHECK-NOT: math.fma

// The shared multiply may come later; its operands are the fused operation's
// own, so it only has to move up.
func.func @relaxes_when_the_product_comes_later(%a: f32, %b: f32, %c: f32) -> (f32, f32) {
  %0 = math.fma %a, %b, %c {ondsp.fast_used = ["fuse_multiply_add"]} : f32
  %p = arith.mulf %b, %a : f32
  return %0, %p : f32, f32
}

// CHECK-LABEL: func.func @relaxes_when_the_product_comes_later
// CHECK: %[[P:.*]] = arith.mulf
// CHECK: arith.addf %{{.*}}, %[[P]]
// CHECK-NOT: math.fma

// Under the `fma` contract the fused form IS the declared numeric, so no
// permission was spent and nothing may be relaxed.
func.func @leaves_the_declared_fma_alone(%a: f32, %b: f32, %c: f32) -> (f32, f32) {
  %p = arith.mulf %a, %b : f32
  %0 = math.fma %a, %b, %c : f32
  return %0, %p : f32, f32
}

// CHECK-LABEL: func.func @leaves_the_declared_fma_alone
// CHECK: math.fma
