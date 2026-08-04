; REQUIRES: x86-registered-target
; RUN: llc --version | FileCheck %s --check-prefix=TOOLCHAIN
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=-fma,-fma4 %S/Inputs/fma_flag_matrix.ll -o - | FileCheck %s --check-prefix=NOFMA
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64 -mattr=+fma %S/Inputs/fma_flag_matrix.ll -o - | FileCheck %s --check-prefix=X86FMA --implicit-check-not=mulss

; The one configuration measured to expand llvm.fma, which LangRef specifies as
; a single fused event. Rationale: consumeFastPermission in OndspSemantics.h.
; Version, cpu and mattr are pinned because another toolchain has to be
; measured again. Companions: _aarch64 and _arm.

; TOOLCHAIN: LLVM version 17.0.6

; NOFMA-LABEL: unflagged:
; NOFMA: fmaf
; X86FMA-LABEL: unflagged:
; X86FMA: vfmadd

; contract alone never licenses de-fusion.
; NOFMA-LABEL: contract_only:
; NOFMA: fmaf
; X86FMA-LABEL: contract_only:
; X86FMA: vfmadd

; reassoc does, and only here.
; NOFMA-LABEL: reassoc_only:
; NOFMA: mulss
; NOFMA: addss
; X86FMA-LABEL: reassoc_only:
; X86FMA: vfmadd

; NOFMA-LABEL: reassoc_contract:
; NOFMA: mulss
; NOFMA: addss
; X86FMA-LABEL: reassoc_contract:
; X86FMA: vfmadd
