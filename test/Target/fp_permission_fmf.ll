; RUN: llc --version | FileCheck %s --check-prefix=TOOLCHAIN
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=-fma,-fma4 %s -o - | FileCheck %s --check-prefix=NOFMA
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=+fma %s -o - | FileCheck %s --check-prefix=X86FMA --implicit-check-not=mulss
; RUN: llc -mtriple=aarch64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=A64 --implicit-check-not=fmul
; RUN: llc -mtriple=thumbv7em-none-eabi -mcpu=cortex-m4 -float-abi=hard %s -o - | FileCheck %s --check-prefix=M4F --implicit-check-not=vmul
; RUN: llc -mtriple=thumbv7m-none-eabi -mcpu=cortex-m3 %s -o - | FileCheck %s --check-prefix=M3 --implicit-check-not=vmul
; RUN: llc -mtriple=armv7a-none-eabi -mcpu=cortex-a9 -float-abi=hard %s -o - | FileCheck %s --check-prefix=A9 --implicit-check-not=vmul

; Measured, not derived: LangRef specifies llvm.fma as one fused event and one
; of these configurations expands it anyway.
;
; contract never de-fuses. reassoc de-fuses on the X86 backend without +fma and
; nowhere else measured - not on AArch64, and not on the 32-bit ARM DSP targets
; even where there is no fused instruction and the fallback is a libm call. So
; the hazard is a per-backend expansion policy, not a target-capability rule,
; and a delegated permission cannot be reasoned about from the target alone.
; Rationale: consumeFastPermission in OndspSemantics.h.

; TOOLCHAIN: LLVM version 17.0.6

declare float @llvm.fma.f32(float, float, float)

; NOFMA-LABEL: unflagged:
; NOFMA: fmaf
; X86FMA-LABEL: unflagged:
; X86FMA: vfmadd
; A64-LABEL: unflagged:
; A64: fmadd
; M4F-LABEL: unflagged:
; M4F: vfma.f32
; M3-LABEL: unflagged:
; M3: bl{{.*}}fmaf
; A9-LABEL: unflagged:
; A9: {{b|bl}}{{.*}}fmaf
define float @unflagged(float %a, float %b, float %c) {
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; contract alone never licenses de-fusion anywhere measured.
; NOFMA-LABEL: contract_only:
; NOFMA: fmaf
; X86FMA-LABEL: contract_only:
; X86FMA: vfmadd
; A64-LABEL: contract_only:
; A64: fmadd
; M4F-LABEL: contract_only:
; M4F: vfma.f32
; M3-LABEL: contract_only:
; M3: bl{{.*}}fmaf
; A9-LABEL: contract_only:
; A9: {{b|bl}}{{.*}}fmaf
define float @contract_only(float %a, float %b, float %c) {
  %r = call contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; reassoc does, on the X86 backend without +fma and nowhere else.
; NOFMA-LABEL: reassoc_only:
; NOFMA: mulss
; NOFMA: addss
; X86FMA-LABEL: reassoc_only:
; X86FMA: vfmadd
; A64-LABEL: reassoc_only:
; A64: fmadd
; M4F-LABEL: reassoc_only:
; M4F: vfma.f32
; M3-LABEL: reassoc_only:
; M3: bl{{.*}}fmaf
; A9-LABEL: reassoc_only:
; A9: {{b|bl}}{{.*}}fmaf
define float @reassoc_only(float %a, float %b, float %c) {
  %r = call reassoc float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

; NOFMA-LABEL: reassoc_contract:
; NOFMA: mulss
; NOFMA: addss
; X86FMA-LABEL: reassoc_contract:
; X86FMA: vfmadd
; A64-LABEL: reassoc_contract:
; A64: fmadd
; M4F-LABEL: reassoc_contract:
; M4F: vfma.f32
; M3-LABEL: reassoc_contract:
; M3: bl{{.*}}fmaf
; A9-LABEL: reassoc_contract:
; A9: {{b|bl}}{{.*}}fmaf
define float @reassoc_contract(float %a, float %b, float %c) {
  %r = call reassoc contract float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}
