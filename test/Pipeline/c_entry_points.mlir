// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=llvm | FileCheck %s
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=c-header | FileCheck %s --check-prefix=HEADER
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot.ox --emit=llvm | ondrix-translate - --mlir-to-ondrix-c-header | FileCheck %s --check-prefix=HEADER
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_constexpr.ox --emit=c-header | FileCheck %s --check-prefix=STATIC
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_gain.ox --emit=llvm | FileCheck %s --check-prefix=GAIN
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_gain.ox --emit=c-header | FileCheck %s --check-prefix=GAIN-HEADER
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_matmul_floor.ox --emit=c-header | FileCheck %s --check-prefix=MATRIX
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_lms_floor.ox --emit=c-header | FileCheck %s --check-prefix=PAIR
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_filter_valid.ox --emit=c-header | FileCheck %s --check-prefix=NONE

// The frontend declares the two windows equal, so the entry takes one length
// in the host index width and hands both descriptors the same size.
// CHECK-LABEL: llvm.func @ondrix_q15_dot(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr, %[[B:.*]]: !llvm.ptr, %[[N:.*]]: i64) -> i16
// CHECK: llvm.call @q15_dot(%[[A]], %[[A]], %{{.*}}, %[[N]], %{{.*}}, %[[B]], %[[B]], %{{.*}}, %[[N]], %{{.*}})
// HEADER: int16_t ondrix_q15_dot(const int16_t *lhs, const int16_t *rhs, uint64_t length);
// STATIC: int16_t ondrix_q15_fir_constexpr(const int16_t *window);

// A static tensor result is written into the caller's buffer: the kernel takes
// it as a trailing noalias destination and allocates nothing.
// GAIN-LABEL: llvm.func @f32_gain(
// GAIN-SAME: %{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: !llvm.ptr {llvm.noalias}, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
// GAIN-NOT: @malloc
// GAIN-LABEL: llvm.func @ondrix_f32_gain(
// GAIN-SAME: %[[IN:.*]]: !llvm.ptr, %[[OUT:.*]]: !llvm.ptr)
// GAIN: llvm.call @f32_gain(%[[IN]], %[[IN]], %{{.*}}, %{{.*}}, %{{.*}}, %[[OUT]], %[[OUT]], %{{.*}}, %{{.*}}, %{{.*}})
// GAIN-NEXT: llvm.return
// GAIN-HEADER: void ondrix_f32_gain(const float *input, float *output);
// MATRIX: void ondrix_q15_matmul_floor(const int16_t *a, const int16_t *b, int16_t *output);
// PAIR: void ondrix_q31_lms_floor(const int32_t *x, const int32_t *d, const int32_t *w, int32_t *output0, int32_t *output1);
// NONE-NOT: ondrix_q15_fir_filter
