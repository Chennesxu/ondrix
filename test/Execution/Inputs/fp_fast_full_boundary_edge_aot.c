#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* A full-boundary FIR at K = 20 and N = 32 with W = 8 splits three ways: the
 * two guarded edge ranges and an interior that reaches the horizontal rebuild.
 * The declared contract says an out-of-range tap performs NO accumulator
 * update, which is not the same as contributing a zero term - for finite
 * values the two agree, so an infinity is what separates them. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32;

extern void _mlir_ciface_f32_fast_full(MemRefF32 *, MemRefF32 *, MemRefF32 *, MemRefF32 *);

enum { kInputLength = 32, kTapCount = 20, kOutputLength = kInputLength + kTapCount - 1 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int expectBits(const char *label, float got, float want) {
  if (floatBits(got) == floatBits(want))
    return 0;
  fprintf(stderr, "%s: got %a (0x%08x), expected %a (0x%08x)\n", label, (double)got, floatBits(got),
          (double)want, floatBits(want));
  return 1;
}

static int expectNan(const char *label, float got) {
  if (isnan(got))
    return 0;
  fprintf(stderr, "%s: got %a (0x%08x), expected a NaN\n", label, (double)got, floatBits(got));
  return 1;
}

int main(void) {
  float input[kInputLength];
  float coefficients[kTapCount];
  float output[kOutputLength];

  for (int64_t i = 0; i < kInputLength; ++i)
    input[i] = 1.0f;
  input[0] = 0.5f;
  input[kInputLength - 1] = 0.25f;

  for (int64_t k = 0; k < kTapCount; ++k)
    coefficients[k] = 1.0f;
  coefficients[0] = 5.0f;
  coefficients[1] = INFINITY;
  coefficients[2] = NAN;
  coefficients[kTapCount - 1] = 3.0f;

  for (int64_t i = 0; i < kOutputLength; ++i)
    output[i] = 0.0f;

  MemRefF32 inputRef = {input, input, 0, {kInputLength}, {1}};
  MemRefF32 coefficientRef = {coefficients, coefficients, 0, {kTapCount}, {1}};
  MemRefF32 outputRef = {output, output, 0, {kOutputLength}, {1}};
  MemRefF32 resultRef;
  _mlir_ciface_f32_fast_full(&resultRef, &inputRef, &coefficientRef, &outputRef);

  /* Output 0 pairs coefficient 19 with input 0 and skips taps 0..18, so the
   * infinity and the NaN are never evaluated. Zero padding would reach
   * 0 * Inf and return a NaN. Output 50 is the mirror case: only tap 0. */
  int failed = expectBits("left edge, all poison taps skipped", output[0], 1.5f);
  failed |= expectBits("right edge, all poison taps skipped", output[kOutputLength - 1], 1.25f);

  /* Non-vacuity. Output 18 is still an edge window but taps 1 and 2 are in
   * range there, and output 19 is the first interior window, so both must be
   * NaN: the poison is genuinely in the corpus and both routes evaluate it. */
  failed |= expectNan("edge window that does reach the poison", output[18]);
  failed |= expectNan("interior window that does reach the poison", output[19]);

  return failed;
}
