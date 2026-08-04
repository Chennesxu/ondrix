; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-unknown-linux-gnu %S/Inputs/fma_flag_matrix.ll -o - | FileCheck %s --implicit-check-not=fmul

; No flag combination de-fuses here, so the X86 expansion measured in
; _x86 is a backend policy rather than a target-capability rule.

; CHECK-LABEL: unflagged:
; CHECK: fmadd
; CHECK-LABEL: contract_only:
; CHECK: fmadd
; CHECK-LABEL: reassoc_only:
; CHECK: fmadd
; CHECK-LABEL: reassoc_contract:
; CHECK: fmadd
