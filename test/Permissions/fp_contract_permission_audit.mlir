// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce --lower-ondsp-f32-reduce-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | ondrix-translate --mlir-to-llvmir | FileCheck %s --implicit-check-not=nnan --implicit-check-not=ninf --implicit-check-not=nsz --implicit-check-not=arcp --implicit-check-not=afn --implicit-check-not="fast float" --implicit-check-not="fast <8 x float>"

// The permission audit runs on the translated LLVM IR because that is where
// this flow's floating-point permissions are final: nothing between this
// point and the object re-runs an IR optimization that could widen them.
// The implicit-check-not list is the escalation set no contract declares —
// including LLVM's blanket `fast` spelling, which stands for all of them.

// An off site keeps a separate unflagged multiply and add per element.
// CHECK-LABEL: define float @off_site(
// CHECK: = fmul float %
// CHECK: = fadd float %

// An fma site is an explicit fused event, not a contraction permission: the
// intrinsic call carries no flag at all.
// CHECK-LABEL: define float @fma_site(
// CHECK: = call float @llvm.fma.f32(

// A fast site carries exactly the two declared permissions. The cross-lane
// fold is the ordered intrinsic with an explicit +0.0 start, so the
// reassociation permission is spent on the lane partition and nowhere else.
// CHECK-LABEL: define float @fast_site(
// CHECK: = call reassoc contract <8 x float> @llvm.fma.v8f32(
// CHECK: = call float @llvm.vector.reduce.fadd.v8f32(float 0.000000e+00, <8 x float>
// CHECK: = call reassoc contract float @llvm.fma.f32(
// CHECK: = fadd reassoc contract float

// Four sites in one function: permissions must not bleed across call sites
// in either direction, and neighbouring plain arithmetic must stay unflagged.
// CHECK-LABEL: define float @mixed_contract_sites(
// CHECK: = fmul float %
// CHECK: = fadd float %
// CHECK: = call float @llvm.fma.f32(
// CHECK: = call reassoc contract <8 x float> @llvm.fma.v8f32(
// CHECK: = fmul float %
// CHECK: = fadd float %
// CHECK: = fadd float %

func.func @off_site(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

func.func @fma_site(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

func.func @fast_site(%lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %r = ondsp.reduce_mac %zero, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

func.func @mixed_contract_sites(%a: memref<?xf32>, %b: memref<?xf32>) -> f32 {
  %zero = arith.constant 0.0 : f32
  %off0 = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  %fma = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fp<format = f32, contract = fma>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  %fst = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  %off1 = ondsp.reduce_mac %zero, %a, %b {numeric = #ondsp.fp<format = f32, contract = off>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  %s0 = arith.addf %off0, %fma : f32
  %s1 = arith.addf %s0, %fst : f32
  %s2 = arith.addf %s1, %off1 : f32
  return %s2 : f32
}
