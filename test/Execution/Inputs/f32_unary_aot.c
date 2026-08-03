#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the f32 moving average and DCT. Both are exact contracts,
 * so every comparison is bit for bit against a reference that recomputes the
 * declared event graph, including its own cosine table. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32;

extern void _mlir_ciface_f32_moving_average_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_dct_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_dct_fma(MemRefF32 *, MemRefF32 *);

enum { kLength = 8, kWindow = 3, kAverages = 6, kTrialCount = 32 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float referenceAverage(const float *x, int64_t n) {
  float sum = x[n];
  for (int64_t k = 1; k < kWindow; ++k)
    sum = sum + x[n + k];
  return sum / (float)kWindow;
}

/* The reference derives the table itself, in double, and rounds once to f32 —
 * a compiler-side table change breaks this gate. */
static float referenceDct(const float *x, int64_t k, int fused) {
  float sum = 0.0f;
  for (int64_t n = 0; n < kLength; ++n) {
    const double angle = 3.14159265358979323846 * (double)((2 * n + 1) * k) / (2.0 * kLength);
    const float coefficient = (float)cos(angle);
    if (n == 0)
      sum = x[n] * coefficient;
    else
      sum = fused ? fmaf(x[n], coefficient, sum) : sum + x[n] * coefficient;
  }
  return sum;
}

static int check(const float *x, const char *label) {
  float input[kLength];
  memcpy(input, x, sizeof(input));
  MemRefF32 inputRef = {input, input, 0, {kLength}, {1}};
  MemRefF32 average, dctOff, dctFma;
  _mlir_ciface_f32_moving_average_off(&average, &inputRef);
  _mlir_ciface_f32_dct_off(&dctOff, &inputRef);
  _mlir_ciface_f32_dct_fma(&dctFma, &inputRef);

  int failed = 0;
  for (int64_t n = 0; n < kAverages; ++n) {
    const float expected = referenceAverage(x, n);
    const float actual = average.aligned[average.offset + n * average.strides[0]];
    if (floatBits(actual) != floatBits(expected)) {
      fprintf(stderr, "%s average [%lld]: got %a, expected %a\n", label, (long long)n,
              (double)actual, (double)expected);
      failed = 1;
    }
  }
  for (int64_t k = 0; k < kLength; ++k) {
    const float expectedOff = referenceDct(x, k, 0);
    const float expectedFma = referenceDct(x, k, 1);
    const float actualOff = dctOff.aligned[dctOff.offset + k * dctOff.strides[0]];
    const float actualFma = dctFma.aligned[dctFma.offset + k * dctFma.strides[0]];
    if (floatBits(actualOff) != floatBits(expectedOff)) {
      fprintf(stderr, "%s dct off [%lld]: got %a, expected %a\n", label, (long long)k,
              (double)actualOff, (double)expectedOff);
      failed = 1;
    }
    if (floatBits(actualFma) != floatBits(expectedFma)) {
      fprintf(stderr, "%s dct fma [%lld]: got %a, expected %a\n", label, (long long)k,
              (double)actualFma, (double)expectedFma);
      failed = 1;
    }
  }
  free(average.allocated);
  free(dctOff.allocated);
  free(dctFma.allocated);
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

int main(void) {
  float x[kLength];
  for (int64_t i = 0; i < kLength; ++i)
    x[i] = 0.0f;
  int failed = check(x, "zero");

  /* A residual at the 2^-24 tie that a fused DCT term keeps and a rounded
   * product loses. */
  x[0] = 1.0f;
  x[1] = 0x1.000002p-12f;
  for (int64_t i = 2; i < kLength; ++i)
    x[i] = 0.0f;
  failed |= check(x, "contract split");

  uint32_t state = UINT32_C(0x1F123BB5);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    snprintf(label, sizeof label, "trial %d", trial);
    for (int64_t i = 0; i < kLength; ++i)
      x[i] = (float)(int16_t)(nextRandom(&state) >> 16) / 8192.0f;
    failed |= check(x, label);
  }
  return failed;
}
