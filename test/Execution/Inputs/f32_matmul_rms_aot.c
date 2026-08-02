#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the f32 matmul and rms contracts. Both are exact, so every
 * comparison here is bit for bit against a reference that computes the
 * declared event graph itself. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRefF32Rank2;

extern void _mlir_ciface_f32_matmul_off(MemRefF32Rank2 *, MemRefF32Rank2 *, MemRefF32Rank2 *);
extern void _mlir_ciface_f32_matmul_fma(MemRefF32Rank2 *, MemRefF32Rank2 *, MemRefF32Rank2 *);
extern void _mlir_ciface_f32_matmul(MemRefF32Rank2 *, MemRefF32Rank2 *, MemRefF32Rank2 *);
extern void _mlir_ciface_f32_rms_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_rms_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);

enum { kRows = 3, kInner = 4, kColumns = 2, kRmsLength = 10, kTrialCount = 32 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* C[i][j] over increasing k from +0.0: one multiply and one add per term
 * under off, one fused event per term under fma. */
static float referenceMatmulElement(const float *a, const float *b, int64_t row, int64_t column,
                                    int fused) {
  float accumulator = 0.0f;
  for (int64_t k = 0; k < kInner; ++k) {
    const float left = a[row * kInner + k];
    const float right = b[k * kColumns + column];
    accumulator = fused ? fmaf(left, right, accumulator) : accumulator + left * right;
  }
  return accumulator;
}

/* The reduction is contract indexed; the mean and the root are the same
 * single operations in every mode. */
static float referenceRms(const float *x, int fused) {
  float sumsq = 0.0f;
  for (int64_t i = 0; i < kRmsLength; ++i)
    sumsq = fused ? fmaf(x[i], x[i], sumsq) : sumsq + x[i] * x[i];
  return sqrtf(sumsq / (float)kRmsLength);
}

static int checkMatmul(const float *a, const float *b, const char *label) {
  float lhs[kRows * kInner];
  float rhs[kInner * kColumns];
  memcpy(lhs, a, sizeof(lhs));
  memcpy(rhs, b, sizeof(rhs));

  MemRefF32Rank2 lhsRef = {lhs, lhs, 0, {kRows, kInner}, {kInner, 1}};
  MemRefF32Rank2 rhsRef = {rhs, rhs, 0, {kInner, kColumns}, {kColumns, 1}};
  MemRefF32Rank2 off, fma, source;
  _mlir_ciface_f32_matmul_off(&off, &lhsRef, &rhsRef);
  _mlir_ciface_f32_matmul_fma(&fma, &lhsRef, &rhsRef);
  _mlir_ciface_f32_matmul(&source, &lhsRef, &rhsRef);

  int failed = 0;
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t column = 0; column < kColumns; ++column) {
      const int64_t index = row * off.strides[0] + column * off.strides[1] + off.offset;
      const float expectedOff = referenceMatmulElement(a, b, row, column, 0);
      const float expectedFma = referenceMatmulElement(a, b, row, column, 1);
      if (floatBits(off.aligned[index]) != floatBits(expectedOff)) {
        fprintf(stderr, "%s matmul off [%lld][%lld]: got %a, expected %a\n", label, (long long)row,
                (long long)column, (double)off.aligned[index], (double)expectedOff);
        failed = 1;
      }
      if (floatBits(fma.aligned[index]) != floatBits(expectedFma)) {
        fprintf(stderr, "%s matmul fma [%lld][%lld]: got %a, expected %a\n", label, (long long)row,
                (long long)column, (double)fma.aligned[index], (double)expectedFma);
        failed = 1;
      }
      /* The .ox binding declares fma, so the source-level object must agree
       * with the hand-written one element for element. */
      if (floatBits(source.aligned[index]) != floatBits(expectedFma)) {
        fprintf(stderr, "%s matmul .ox [%lld][%lld]: got %a, expected %a\n", label, (long long)row,
                (long long)column, (double)source.aligned[index], (double)expectedFma);
        failed = 1;
      }
    }
  }
  free(off.allocated);
  free(fma.allocated);
  free(source.allocated);
  return failed;
}

static int checkRms(const float *x, const char *label) {
  float input[kRmsLength];
  memcpy(input, x, sizeof(input));

  MemRefF32Rank1 inputRef = {input, input, 0, {kRmsLength}, {1}};
  MemRefF32Rank1 off, fma;
  _mlir_ciface_f32_rms_off(&off, &inputRef);
  _mlir_ciface_f32_rms_fma(&fma, &inputRef);

  int failed = 0;
  const float expectedOff = referenceRms(x, 0);
  const float expectedFma = referenceRms(x, 1);
  if (floatBits(off.aligned[off.offset]) != floatBits(expectedOff)) {
    fprintf(stderr, "%s rms off: got %a, expected %a\n", label, (double)off.aligned[off.offset],
            (double)expectedOff);
    failed = 1;
  }
  if (floatBits(fma.aligned[fma.offset]) != floatBits(expectedFma)) {
    fprintf(stderr, "%s rms fma: got %a, expected %a\n", label, (double)fma.aligned[fma.offset],
            (double)expectedFma);
    failed = 1;
  }
  free(off.allocated);
  free(fma.allocated);
  return failed;
}

static uint32_t nextRandom(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

static float randomValue(uint32_t *state) {
  const int16_t raw = (int16_t)(nextRandom(state) >> 16);
  return (float)raw / 8192.0f;
}

/* The residual of 1 + 2^-23 - 1 survives a fused update and is lost when the
 * product is rounded on its own, so a lowering that fuses an off term or
 * splits an fma term changes exported bits here. */
static int checkContractSplit(void) {
  float a[kRows * kInner] = {0};
  float b[kInner * kColumns] = {0};
  float x[kRmsLength] = {0};

  a[0] = 1.0f;
  b[0] = 0x1.000002p+0f;
  a[1] = -1.0f;
  b[kColumns] = 1.0f;
  int failed = checkMatmul(a, b, "contract split");

  x[0] = 0x1.0p+12f;
  x[1] = 0x1.000002p-12f;
  failed |= checkRms(x, "contract split");
  return failed;
}

int main(void) {
  int failed = checkContractSplit();

  float a[kRows * kInner];
  float b[kInner * kColumns];
  float x[kRmsLength];
  uint32_t state = UINT32_C(0x2545F491);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    snprintf(label, sizeof label, "trial %d", trial);
    for (int i = 0; i < kRows * kInner; ++i)
      a[i] = randomValue(&state);
    for (int i = 0; i < kInner * kColumns; ++i)
      b[i] = randomValue(&state);
    for (int i = 0; i < kRmsLength; ++i)
      x[i] = randomValue(&state);
    failed |= checkMatmul(a, b, label);
    failed |= checkRms(x, label);
  }
  return failed;
}
