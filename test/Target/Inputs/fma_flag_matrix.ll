; Four llvm.fma.f32 sites differing only in fast-math flags. Shared by the
; per-target characterization tests in the parent directory.

declare float @llvm.fma.f32(float, float, float)

define float @unflagged(float %a, float %b, float %c) {
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define float @contract_only(float %a, float %b, float %c) {
  %r = call contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define float @reassoc_only(float %a, float %b, float %c) {
  %r = call reassoc float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define float @reassoc_contract(float %a, float %b, float %c) {
  %r = call reassoc contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}
