; RUN: llc --version | FileCheck %s --check-prefix=TOOLCHAIN
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=-fma,-fma4 %s -o - | FileCheck %s --check-prefix=NOFMA
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=+fma %s -o - | FileCheck %s --check-prefix=X86FMA --implicit-check-not=mulss
; RUN: llc -mtriple=aarch64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=ARM --implicit-check-not=fmul

; Measured, not derived: LangRef specifies llvm.fma as one fused event and one
; of these configurations expands it anyway. Version, triple, cpu and mattr are
; all pinned because another toolchain has to be measured again.
;
; De-fusion needs both reassoc and a target with no fused instruction, so a
; delegated permission makes the realized graph target dependent. Rationale:
; consumeFastPermission in OndspSemantics.h.

; TOOLCHAIN: LLVM version 17.0.6

declare float @llvm.fma.f32(float, float, float)

; NOFMA-LABEL: unflagged:
; NOFMA: fmaf
; X86FMA-LABEL: unflagged:
; X86FMA: vfmadd
; ARM-LABEL: unflagged:
; ARM: fmadd
define float @unflagged(float %a, float %b, float %c) {
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; contract alone never licenses de-fusion, on any of the three.
; NOFMA-LABEL: contract_only:
; NOFMA: fmaf
; X86FMA-LABEL: contract_only:
; X86FMA: vfmadd
; ARM-LABEL: contract_only:
; ARM: fmadd
define float @contract_only(float %a, float %b, float %c) {
  %r = call contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; reassoc does, but only where there is no fused instruction to keep.
; NOFMA-LABEL: reassoc_only:
; NOFMA: mulss
; NOFMA: addss
; X86FMA-LABEL: reassoc_only:
; X86FMA: vfmadd
; ARM-LABEL: reassoc_only:
; ARM: fmadd
define float @reassoc_only(float %a, float %b, float %c) {
  %r = call reassoc float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; NOFMA-LABEL: reassoc_contract:
; NOFMA: mulss
; NOFMA: addss
; X86FMA-LABEL: reassoc_contract:
; X86FMA: vfmadd
; ARM-LABEL: reassoc_contract:
; ARM: fmadd
define float @reassoc_contract(float %a, float %b, float %c) {
  %r = call reassoc contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}
