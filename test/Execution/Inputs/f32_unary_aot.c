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
extern void _mlir_ciface_f32_moving_average_fma(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_moving_average_fast(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_dct_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_dct_fma(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_dct_fast(MemRefF32 *, MemRefF32 *);

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

static int compare(const char *label, const char *mode, int64_t index, float got, float expected) {
  if (floatBits(got) == floatBits(expected))
    return 0;
  fprintf(stderr, "%s %s [%lld]: got %a, expected %a\n", label, mode, (long long)index, (double)got,
          (double)expected);
  return 1;
}

static int check(const float *x, const char *label) {
  float input[kLength];
  memcpy(input, x, sizeof(input));
  MemRefF32 inputRef = {input, input, 0, {kLength}, {1}};
  MemRefF32 average, averageFma, averageFast, dctOff, dctFma, dctFast;
  _mlir_ciface_f32_moving_average_off(&average, &inputRef);
  _mlir_ciface_f32_moving_average_fma(&averageFma, &inputRef);
  _mlir_ciface_f32_moving_average_fast(&averageFast, &inputRef);
  _mlir_ciface_f32_dct_off(&dctOff, &inputRef);
  _mlir_ciface_f32_dct_fma(&dctFma, &inputRef);
  _mlir_ciface_f32_dct_fast(&dctFast, &inputRef);

  int failed = 0;
  /* One window sum in declared order and one division. The window sum IS a
   * reduction tree, so fast's reassociation permission applies, but at K = 3
   * the chained rebuild refuses (its tree is the declared left fold), so all
   * three objects run the declared association. */
  for (int64_t n = 0; n < kAverages; ++n) {
    const float expected = referenceAverage(x, n);
    failed |= compare(label, "average off", n,
                      average.aligned[average.offset + n * average.strides[0]], expected);
    failed |= compare(label, "average fma", n,
                      averageFma.aligned[averageFma.offset + n * averageFma.strides[0]], expected);
    failed |=
        compare(label, "average fast", n,
                averageFast.aligned[averageFast.offset + n * averageFast.strides[0]], expected);
  }
  for (int64_t k = 0; k < kLength; ++k) {
    const float expectedOff = referenceDct(x, k, 0);
    const float expectedFma = referenceDct(x, k, 1);
    failed |= compare(label, "dct off", k, dctOff.aligned[dctOff.offset + k * dctOff.strides[0]],
                      expectedOff);
    failed |= compare(label, "dct fma", k, dctFma.aligned[dctFma.offset + k * dctFma.strides[0]],
                      expectedFma);
    /* The row stays ordered under fast, so the selected member is the fused
     * one. Pinning it pins the selection, not the contract. */
    failed |= compare(label, "dct fast", k,
                      dctFast.aligned[dctFast.offset + k * dctFast.strides[0]], expectedFma);
  }
  free(average.allocated);
  free(averageFma.allocated);
  free(averageFast.allocated);
  free(dctOff.allocated);
  free(dctFma.allocated);
  free(dctFast.allocated);
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

/* The window sum's association is observable, so "no permission consumed" has
 * to be gated rather than asserted. With 1e8 and -1e8 adjacent, the declared
 * (x0 + x1) + x2 keeps x2 while x0 + (x1 + x2) loses it under the 1e8 ulp of
 * eight; the harness checks that the two associations really differ before
 * requiring all three objects to run the declared one. */
static int checkWindowAssociation(void) {
  float x[kLength];
  for (int64_t i = 0; i < kLength; ++i)
    x[i] = 0.0f;
  x[0] = 1.0e8f;
  x[1] = -1.0e8f;
  x[2] = 1.0f;

  const float declared = ((x[0] + x[1]) + x[2]) / (float)kWindow;
  const float regrouped = (x[0] + (x[1] + x[2])) / (float)kWindow;
  if (floatBits(declared) == floatBits(regrouped)) {
    fprintf(stderr, "window association corpus is vacuous: both groupings agree\n");
    return 1;
  }
  return check(x, "window association");
}

int main(void) {
  float x[kLength];
  for (int64_t i = 0; i < kLength; ++i)
    x[i] = 0.0f;
  int failed = check(x, "zero");

  /* Row 0 of the DCT has all-positive coefficients, so an all -0.0 input keeps
   * a -0.0 sum only if each row starts AT its first product; a row seeded with
   * the additive identity would export +0.0. */
  for (int64_t i = 0; i < kLength; ++i)
    x[i] = -0.0f;
  failed |= check(x, "negative zero");

  /* A residual at the 2^-24 tie that a fused DCT term keeps and a rounded
   * product loses. */
  x[0] = 1.0f;
  x[1] = 0x1.000002p-12f;
  for (int64_t i = 2; i < kLength; ++i)
    x[i] = 0.0f;
  failed |= check(x, "contract split");

  failed |= checkWindowAssociation();

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
