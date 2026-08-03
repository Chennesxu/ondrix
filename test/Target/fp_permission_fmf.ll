; RUN: llc --version | FileCheck %s --check-prefix=TOOLCHAIN
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=-fma,-fma4 %s -o - | FileCheck %s

; Toolchain characterization, not a LangRef consequence. LangRef specifies
; llvm.fma as one fused event; this backend nonetheless expands it under
; reassoc, and that is why Ondrix emits no permission it does not intend the
; backend to act on. Everything the measurement depends on is pinned here,
; because a different toolchain has to be measured again rather than assumed.

; TOOLCHAIN: LLVM version 17.0.6

declare float @llvm.fma.f32(float, float, float)

; CHECK-LABEL: unflagged:
; CHECK: fmaf
define float @unflagged(float %a, float %b, float %c) {
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; contract alone does not license de-fusion here.
; CHECK-LABEL: contract_only:
; CHECK: fmaf
define float @contract_only(float %a, float %b, float %c) {
  %r = call contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; reassoc does, with or without contract. This is the flag that would have let
; the backend pick the other member of a fast site's legal set.
; CHECK-LABEL: reassoc_only:
; CHECK: mulss
; CHECK: addss
define float @reassoc_only(float %a, float %b, float %c) {
  %r = call reassoc float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; CHECK-LABEL: reassoc_contract:
; CHECK: mulss
; CHECK: addss
define float @reassoc_contract(float %a, float %b, float %c) {
  %r = call reassoc contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}
