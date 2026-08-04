; REQUIRES: arm-registered-target
; RUN: llc -mtriple=thumbv7em-none-eabi -mcpu=cortex-m4 -float-abi=hard %S/Inputs/fma_flag_matrix.ll -o - | FileCheck %s --check-prefix=M4F --implicit-check-not=vmul
; RUN: llc -mtriple=thumbv7m-none-eabi -mcpu=cortex-m3 %S/Inputs/fma_flag_matrix.ll -o - | FileCheck %s --check-prefix=M3 --implicit-check-not=vmul
; RUN: llc -mtriple=armv7a-none-eabi -mcpu=cortex-a9 -float-abi=hard %S/Inputs/fma_flag_matrix.ll -o - | FileCheck %s --check-prefix=A9 --implicit-check-not=vmul

; The DSP-relevant configurations: Cortex-M is 32-bit only and is where
; embedded DSP runs. M3 and A9 have no fused instruction and still do not
; expand under reassoc, which is what rules out the capability explanation.

; M4F-LABEL: unflagged:
; M4F: vfma.f32
; M3-LABEL: unflagged:
; M3: bl{{.*}}fmaf
; A9-LABEL: unflagged:
; A9: {{b|bl}}{{.*}}fmaf

; M4F-LABEL: contract_only:
; M4F: vfma.f32
; M3-LABEL: contract_only:
; M3: bl{{.*}}fmaf
; A9-LABEL: contract_only:
; A9: {{b|bl}}{{.*}}fmaf

; M4F-LABEL: reassoc_only:
; M4F: vfma.f32
; M3-LABEL: reassoc_only:
; M3: bl{{.*}}fmaf
; A9-LABEL: reassoc_only:
; A9: {{b|bl}}{{.*}}fmaf

; M4F-LABEL: reassoc_contract:
; M4F: vfma.f32
; M3-LABEL: reassoc_contract:
; M3: bl{{.*}}fmaf
; A9-LABEL: reassoc_contract:
; A9: {{b|bl}}{{.*}}fmaf
