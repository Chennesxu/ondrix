// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=llvm | FileCheck %s
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=c-header | FileCheck %s --check-prefix=HEADER
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=llvm | ondrix-translate - --mlir-to-ondrix-c-header | FileCheck %s --check-prefix=HEADER
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_constexpr.ox --emit=c-header | FileCheck %s --check-prefix=STATIC
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_filter_valid.ox --emit=c-header | FileCheck %s --check-prefix=NONE

// The frontend declares the two windows equal, so the entry takes one length
// in the host index width and hands both descriptors the same size.
// CHECK-LABEL: llvm.func @ondrix_q15_dot(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr, %[[B:.*]]: !llvm.ptr, %[[N:.*]]: i64) -> i16
// CHECK: llvm.call @q15_dot(%[[A]], %[[A]], %{{.*}}, %[[N]], %{{.*}}, %[[B]], %[[B]], %{{.*}}, %[[N]], %{{.*}})
// HEADER: int16_t ondrix_q15_dot(const int16_t *lhs, const int16_t *rhs, uint64_t length);
// STATIC: int16_t ondrix_q15_fir_constexpr(const int16_t *window);
// NONE-NOT: ondrix_q15_fir_filter
