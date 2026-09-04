// RUN: ondrix-compile %S/Inputs/f32_fir_filter_valid.ox --emit=llvmir | FileCheck %s
// RUN: ondrix-compile %S/Inputs/f32_fir_filter_valid.ox --emit=llvmir --llvm-opt-level=0 | FileCheck %s --check-prefix=O0
// RUN: not ondrix-compile %S/Inputs/f32_fir_filter_valid.ox --emit=llvmir --llvm-opt-level=4 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: ondrix-compile %S/Inputs/f32_fir_filter_valid.ox --emit=manifest | FileCheck %s --check-prefix=MANIFEST

// LLVM IR, not LLVM dialect, with the middle end already run at the requested
// level; the O3 module carries no fast-math flag.
// CHECK: define {{.*}}@
// CHECK-NOT: llvm.func
// CHECK-NOT: fast
// CHECK-NOT: reassoc
// O0: define {{.*}}@
// BAD: --llvm-opt-level must be 0 to 3
// MANIFEST: "llvm_middle_end"
// MANIFEST-NEXT: "loop_vectorize": false
// MANIFEST-NEXT: "opt_level": 3
// MANIFEST-NEXT: "slp_vectorize": false
