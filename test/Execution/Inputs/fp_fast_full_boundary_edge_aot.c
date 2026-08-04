#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* An out-of-range tap performs no accumulator update, which a materialized zero
 * term differs from in exactly the two ways gated below (ledger: why no third). */

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

/* The propagated value is the poison itself for an infinity and a NaN for a
 * NaN, so reachability asserts non-finite rather than either one. */
static int expectNonFinite(const char *label, float got) {
  if (!isfinite(got))
    return 0;
  fprintf(stderr, "%s: got %a (0x%08x), expected a non-finite value\n", label, (double)got,
          floatBits(got));
  return 1;
}

/* One poison value per trial, so a pass attributes the skip to that value
 * alone rather than to whichever of two was reached first. */
static int nonFiniteTrial(const char *what, float poison) {
  float input[kInputLength];
  float coefficients[kTapCount];
  float output[kOutputLength];
  char label[80];

  for (int64_t i = 0; i < kInputLength; ++i)
    input[i] = 1.0f;
  input[0] = 0.5f;
  input[kInputLength - 1] = 0.25f;

  for (int64_t k = 0; k < kTapCount; ++k)
    coefficients[k] = 1.0f;
  coefficients[0] = 5.0f;
  coefficients[1] = poison;
  coefficients[kTapCount - 1] = 3.0f;

  for (int64_t i = 0; i < kOutputLength; ++i)
    output[i] = 0.0f;

  MemRefF32 inputRef = {input, input, 0, {kInputLength}, {1}};
  MemRefF32 coefficientRef = {coefficients, coefficients, 0, {kTapCount}, {1}};
  MemRefF32 outputRef = {output, output, 0, {kOutputLength}, {1}};
  MemRefF32 resultRef;
  _mlir_ciface_f32_fast_full(&resultRef, &inputRef, &coefficientRef, &outputRef);

  /* Output 0 pairs coefficient 19 with input 0 and skips taps 0..18, so the
   * poison is never evaluated. Zero padding reaches 0 * poison and returns
   * NaN. Output 50 is the mirror case: only tap 0 is in range. */
  snprintf(label, sizeof label, "%s: left edge skips the poison tap", what);
  int failed = expectBits(label, output[0], 1.5f);
  snprintf(label, sizeof label, "%s: right edge skips the poison tap", what);
  failed |= expectBits(label, output[kOutputLength - 1], 1.25f);

  /* Non-vacuity. Output 18 is an edge window that does include tap 1, and
   * output 19 is the first interior window, so both must be NaN: the poison is
   * reachable and both routes evaluate it. */
  snprintf(label, sizeof label, "%s: edge window that reaches the poison", what);
  failed |= expectNonFinite(label, output[18]);
  snprintf(label, sizeof label, "%s: interior window that reaches the poison", what);
  failed |= expectNonFinite(label, output[19]);
  return failed;
}

/* The other separator class. With a finite coefficient a materialized zero term
 * can only be a signed zero, so it is invisible against a nonzero accumulator
 * and observable only in the zero's sign. This catches an implementation that
 * guards non-finite coefficients but materializes finite ones, which the trials
 * above cannot reach. N = 1 and K = 2 put a single skipped tap after a valid
 * term whose exact product underflows to -0.0. */
static int signedZeroEdgeTrial(void) {
  float input[1] = {-0x1p-75f};
  float coefficients[2] = {0x1p-75f, 1.0f};
  float output[2] = {0.0f, 0.0f};

  MemRefF32 inputRef = {input, input, 0, {1}, {1}};
  MemRefF32 coefficientRef = {coefficients, coefficients, 0, {2}, {1}};
  MemRefF32 outputRef = {output, output, 0, {2}, {1}};
  MemRefF32 resultRef;
  _mlir_ciface_f32_fast_full(&resultRef, &inputRef, &coefficientRef, &outputRef);

  /* Output 1 takes tap 0 only. fma(-2^-75, 2^-75, +0.0) rounds -2^-150 to
   * -0.0, and skipping tap 1 keeps it. Materializing tap 1 as
   * fma(+0.0, 1.0, -0.0) returns +0.0. */
  return expectBits("right edge keeps the negative zero", output[1], -0.0f);
}

int main(void) {
  int failed = nonFiniteTrial("infinity", INFINITY);
  failed |= nonFiniteTrial("NaN", NAN);
  failed |= signedZeroEdgeTrial();
  return failed;
}
