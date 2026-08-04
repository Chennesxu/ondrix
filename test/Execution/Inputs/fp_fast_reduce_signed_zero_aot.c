#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* N = W, N = W + 1 and N = 2W over an all-negative-zero reduction. Every legal
 * member of the fast set returns -0.0 there, so the sign bit is the whole
 * assertion: a synthesized +0.0 anywhere in the tree flips it. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32;

extern float _mlir_ciface_fast_reduce_8(float, MemRefF32 *, MemRefF32 *);
extern float _mlir_ciface_fast_reduce_9(float, MemRefF32 *, MemRefF32 *);
extern float _mlir_ciface_fast_reduce_16(float, MemRefF32 *, MemRefF32 *);
extern float _mlir_ciface_fast_reduce_dynamic(float, MemRefF32 *, MemRefF32 *);

enum { kMaxLength = 32 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int check(const char *label, int64_t length,
                 float (*kernel)(float, MemRefF32 *, MemRefF32 *)) {
  float lhs[kMaxLength];
  float rhs[kMaxLength];
  for (int64_t i = 0; i < length; ++i) {
    lhs[i] = -0.0f;
    rhs[i] = 1.0f;
  }
  MemRefF32 lhsRef = {lhs, lhs, 0, {length}, {1}};
  MemRefF32 rhsRef = {rhs, rhs, 0, {length}, {1}};
  const float got = kernel(-0.0f, &lhsRef, &rhsRef);
  if (floatBits(got) == UINT32_C(0x80000000))
    return 0;
  fprintf(stderr, "%s: got %a (0x%08x), expected -0.0 (0x80000000)\n", label, (double)got,
          floatBits(got));
  return 1;
}

/* The dynamic kernel takes its extent at runtime, so one object covers the
 * empty reduction, the short branch, the boundary, the tail, and a full loop
 * iteration followed by a tail. */
static int checkDynamic(int64_t length) {
  char label[32];
  snprintf(label, sizeof label, "dynamic N = %lld", (long long)length);
  return check(label, length, _mlir_ciface_fast_reduce_dynamic);
}

int main(void) {
  int failed = check("N = W", 8, _mlir_ciface_fast_reduce_8);
  failed |= check("N = W + 1", 9, _mlir_ciface_fast_reduce_9);
  failed |= check("N = 2W", 16, _mlir_ciface_fast_reduce_16);
  static const int64_t kDynamicLengths[] = {0, 1, 7, 8, 9, 17};
  for (size_t i = 0; i < sizeof kDynamicLengths / sizeof *kDynamicLengths; ++i)
    failed |= checkDynamic(kDynamicLengths[i]);
  return failed;
}
