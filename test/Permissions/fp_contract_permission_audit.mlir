// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce --lower-ondsp-f32-reduce-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | ondrix-translate --mlir-to-llvmir | FileCheck %s --implicit-check-not=reassoc --implicit-check-not=contract --implicit-check-not=nnan --implicit-check-not=ninf --implicit-check-not=nsz --implicit-check-not=arcp --implicit-check-not=afn --implicit-check-not="fast float" --implicit-check-not="fast <8 x float>"
// RUN: ondrix-opt %s --vectorize-ondsp-fp-fast-memref-reduce="supports-vector-fma=true" --lower-ondsp-f32-reduce-to-scalar --convert-scf-to-cf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | ondrix-translate --mlir-to-llvmir | FileCheck %s --check-prefix=FUSED --implicit-check-not=reassoc --implicit-check-not=contract --implicit-check-not=nnan --implicit-check-not=ninf --implicit-check-not=nsz --implicit-check-not=arcp --implicit-check-not=afn --implicit-check-not="fast float"

// The permission audit runs on the translated LLVM IR because that is where
// this flow's floating-point permissions are final: nothing between this
// point and the object re-runs an IR optimization that could widen them.
//
// The implicit-check-not list is the whole vocabulary, not just the escalation
// set: every fast permission is spent by the compiler, so the audit point sees
// the selected graph and no licence to reselect.

// An off site keeps a separate multiply and add per element.
// CHECK-LABEL: define float @off_site(
// CHECK: = fmul float %
// CHECK: = fadd float %

// An fma site is an explicit fused event, not a contraction permission.
// CHECK-LABEL: define float @fma_site(
// CHECK: = call float @llvm.fma.f32(

// A fast site spends the reassociation on the lane partition. Neither the
// lane seed nor the fold start is synthesized: both are data.
// CHECK-LABEL: define float @fast_site(
// CHECK: = fmul <8 x float> %
// CHECK: = fmul <8 x float> %
// CHECK: = fadd <8 x float> %
// CHECK: = call float @llvm.vector.reduce.fadd.v8f32(float %0, <8 x float>
// CHECK: = fmul float %
// CHECK: = fadd float %

// Four sites in one function: permissions must not bleed across call sites
// in either direction, and neighbouring plain arithmetic must stay unflagged.
// The name avoids the word this file greps for, which a symbol would match.
// CHECK-LABEL: define float @mixed_policy_sites(
// CHECK: = fmul float %
// CHECK: = fadd float %
// CHECK: = call float @llvm.fma.f32(
// CHECK: = fmul <8 x float> %
// CHECK: = fmul float %
// CHECK: = fadd float %
// CHECK: = fadd float %

// The other selection has to be audited too: it is the one that emits fused
// events, so it is where a delegated permission would actually appear.
// FUSED-LABEL: define float @fast_site(
// FUSED: = fmul <8 x float> %
// FUSED: = call <8 x float> @llvm.fma.v8f32(
// FUSED: = call float @llvm.vector.reduce.fadd.v8f32(float %0, <8 x float>
// FUSED: = call float @llvm.fma.f32(

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

// The initial arrives as an argument so the cross-lane fold's start value
// cannot be satisfied by a synthesized constant.
func.func @fast_site(%init: f32, %lhs: memref<?xf32>, %rhs: memref<?xf32>) -> f32 {
  %r = ondsp.reduce_mac %init, %lhs, %rhs {numeric = #ondsp.fp<format = f32, contract = fast>} : (f32, memref<?xf32>, memref<?xf32>) -> f32
  return %r : f32
}

func.func @mixed_policy_sites(%a: memref<?xf32>, %b: memref<?xf32>) -> f32 {
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
