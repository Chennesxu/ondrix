#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the f32 decimation and interpolation contracts. Both are
 * exact, so every comparison is bit for bit against a reference that walks
 * the declared index relation itself. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

extern void _mlir_ciface_f32_decimate_off(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_decimate_fma(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_interpolate_off(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_interpolate_fma(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_interpolate_fast(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_fir_decimate(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_fir_interpolate(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);

enum {
  kDecimateInput = 12,
  kDecimateTaps = 5,
  kDecimateOutput = 4,
  kInterpolateInput = 4,
  kInterpolateTaps = 3,
  kInterpolateOutput = 9,
  kFactor = 2,
  kTrialCount = 24
};

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* Output m visits taps in increasing k from +0.0 over the window that starts
 * at m * D. */
static float referenceDecimate(const float *input, const float *coeffs, int64_t output, int fused) {
  float accumulator = 0.0f;
  for (int64_t k = 0; k < kDecimateTaps; ++k) {
    const float sample = input[output * kFactor + k];
    accumulator = fused ? fmaf(sample, coeffs[k], accumulator) : accumulator + sample * coeffs[k];
  }
  return accumulator;
}

/* A tap updates the accumulator only when it lands on a real input sample.
 * Terms that would multiply an inserted zero are never evaluated: that skip
 * is the declared event graph, not zero-product elimination. */
static float referenceInterpolate(const float *input, const float *coeffs, int64_t output,
                                  int fused) {
  float accumulator = 0.0f;
  for (int64_t k = 0; k < kInterpolateTaps; ++k) {
    if (output < k)
      continue;
    const int64_t upsampled = output - k;
    if (upsampled % kFactor != 0 || upsampled / kFactor >= kInterpolateInput)
      continue;
    const float sample = input[upsampled / kFactor];
    accumulator = fused ? fmaf(sample, coeffs[k], accumulator) : accumulator + sample * coeffs[k];
  }
  return accumulator;
}

static int compare(const char *label, const char *mode, int64_t index, float got, float expected) {
  if (floatBits(got) == floatBits(expected))
    return 0;
  fprintf(stderr, "%s %s [%lld]: got %a, expected %a\n", label, mode, (long long)index, (double)got,
          (double)expected);
  return 1;
}

static int checkDecimate(const float *input, const float *coeffs, const char *label) {
  float inputCopy[kDecimateInput];
  float coeffCopy[kDecimateTaps];
  memcpy(inputCopy, input, sizeof(inputCopy));
  memcpy(coeffCopy, coeffs, sizeof(coeffCopy));

  MemRefF32Rank1 inputRef = {inputCopy, inputCopy, 0, {kDecimateInput}, {1}};
  MemRefF32Rank1 coeffRef = {coeffCopy, coeffCopy, 0, {kDecimateTaps}, {1}};
  MemRefF32Rank1 off, fma, source;
  _mlir_ciface_f32_decimate_off(&off, &inputRef, &coeffRef);
  _mlir_ciface_f32_decimate_fma(&fma, &inputRef, &coeffRef);
  _mlir_ciface_f32_fir_decimate(&source, &inputRef, &coeffRef);

  int failed = 0;
  for (int64_t m = 0; m < kDecimateOutput; ++m) {
    const float expectedOff = referenceDecimate(input, coeffs, m, 0);
    const float expectedFma = referenceDecimate(input, coeffs, m, 1);
    failed |= compare(label, "decimate off", m, off.aligned[off.offset + m * off.strides[0]],
                      expectedOff);
    failed |= compare(label, "decimate fma", m, fma.aligned[fma.offset + m * fma.strides[0]],
                      expectedFma);
    /* The .ox binding declares fma. */
    failed |= compare(label, "decimate .ox", m,
                      source.aligned[source.offset + m * source.strides[0]], expectedFma);
  }
  free(off.allocated);
  free(fma.allocated);
  free(source.allocated);
  return failed;
}

static int checkInterpolate(const float *input, const float *coeffs, const char *label) {
  float inputCopy[kInterpolateInput];
  float coeffCopy[kInterpolateTaps];
  memcpy(inputCopy, input, sizeof(inputCopy));
  memcpy(coeffCopy, coeffs, sizeof(coeffCopy));

  MemRefF32Rank1 inputRef = {inputCopy, inputCopy, 0, {kInterpolateInput}, {1}};
  MemRefF32Rank1 coeffRef = {coeffCopy, coeffCopy, 0, {kInterpolateTaps}, {1}};
  MemRefF32Rank1 off, fma, fast, source;
  _mlir_ciface_f32_interpolate_off(&off, &inputRef, &coeffRef);
  _mlir_ciface_f32_interpolate_fma(&fma, &inputRef, &coeffRef);
  _mlir_ciface_f32_interpolate_fast(&fast, &inputRef, &coeffRef);
  _mlir_ciface_f32_fir_interpolate(&source, &inputRef, &coeffRef);

  int failed = 0;
  for (int64_t m = 0; m < kInterpolateOutput; ++m) {
    const float expectedOff = referenceInterpolate(input, coeffs, m, 0);
    const float expectedFma = referenceInterpolate(input, coeffs, m, 1);
    failed |= compare(label, "interpolate off", m, off.aligned[off.offset + m * off.strides[0]],
                      expectedOff);
    failed |= compare(label, "interpolate fma", m, fma.aligned[fma.offset + m * fma.strides[0]],
                      expectedFma);
    /* fast spends F here: the fused chain over a rounded product and an add. */
    failed |= compare(label, "interpolate fast", m, fast.aligned[fast.offset + m * fast.strides[0]],
                      expectedFma);
    /* The .ox binding declares off. */
    failed |= compare(label, "interpolate .ox", m,
                      source.aligned[source.offset + m * source.strides[0]], expectedOff);
  }
  free(off.allocated);
  free(fma.allocated);
  free(fast.allocated);
  free(source.allocated);
  return failed;
}

/* Output 1 takes tap 1 only: taps 0 and 2 land on an inserted zero and an
 * out-of-range index. A lowering that materialized those zeros instead of
 * skipping them would be running a different event graph, and these three
 * corpora each expose the difference at that output. */
static int checkNonFiniteTapZero(const char *label, float tap0) {
  float input[kInterpolateInput] = {0.25f, -0.5f, 0.75f, 1.0f};
  float coeffs[kInterpolateTaps] = {tap0, 0.5f, -0.25f};

  MemRefF32Rank1 inputRef = {input, input, 0, {kInterpolateInput}, {1}};
  MemRefF32Rank1 coeffRef = {coeffs, coeffs, 0, {kInterpolateTaps}, {1}};
  MemRefF32Rank1 off;
  _mlir_ciface_f32_interpolate_off(&off, &inputRef, &coeffRef);

  int failed = 0;
  const float skipped = off.aligned[off.offset + off.strides[0]];
  /* Materializing would give 0.0 * tap0, which is NaN for both spellings. */
  if (!isfinite(skipped)) {
    fprintf(stderr, "%s: output 1 is %a, expected a finite value\n", label, (double)skipped);
    failed = 1;
  }
  failed |=
      compare(label, "interpolate off", 1, skipped, referenceInterpolate(input, coeffs, 1, 0));
  free(off.allocated);
  return failed;
}

/* The all-finite separation, which is why the skip cannot be justified as
 * zero-product elimination. Tap 1 is fma(-0x1p-75, 0x1p-75, +0.0): the exact
 * product underflows and the accumulator becomes -0.0. Materializing tap 2's
 * inserted zero would then add +0.0 * 0.5 to it and round back to +0.0. */
static int checkFiniteSignedZeroSkip(void) {
  float input[kInterpolateInput] = {-0x1p-75f, 0.0f, 0.0f, 0.0f};
  float coeffs[kInterpolateTaps] = {0.5f, 0x1p-75f, 0.5f};

  MemRefF32Rank1 inputRef = {input, input, 0, {kInterpolateInput}, {1}};
  MemRefF32Rank1 coeffRef = {coeffs, coeffs, 0, {kInterpolateTaps}, {1}};
  MemRefF32Rank1 fused;
  _mlir_ciface_f32_interpolate_fma(&fused, &inputRef, &coeffRef);

  int failed = 0;
  const float skipped = fused.aligned[fused.offset + fused.strides[0]];
  if (floatBits(skipped) != UINT32_C(0x80000000)) {
    fprintf(stderr, "finite inserted-zero skip: output 1 is %a (0x%08x), expected -0.0\n",
            (double)skipped, floatBits(skipped));
    failed = 1;
  }
  failed |= compare("finite inserted-zero skip", "interpolate fma", 1, skipped,
                    referenceInterpolate(input, coeffs, 1, 1));
  free(fused.allocated);
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

/* A directed split witness: fusing an off term or splitting an fma term
 * changes exported bits here. */
static int checkContractSplit(void) {
  float decimateInput[kDecimateInput] = {0};
  float decimateCoeffs[kDecimateTaps] = {0};
  float interpolateInput[kInterpolateInput] = {0};
  float interpolateCoeffs[kInterpolateTaps] = {0};

  /* Tap 0 leaves the accumulator at -1.0; tap 1 then contributes a product
   * whose exact value needs more than 24 bits, so the cancellation exposes
   * the bits a separate rounding drops. Output 0 of the decimation and
   * output 2 of the interpolation carry the pair. */
  decimateCoeffs[0] = 1.0f;
  decimateInput[0] = -1.0f;
  decimateCoeffs[1] = 0x1.000002p+0f;
  decimateInput[1] = 0x1.000006p+0f;
  int failed = checkDecimate(decimateInput, decimateCoeffs, "contract split");

  interpolateCoeffs[0] = 1.0f;
  interpolateInput[1] = -1.0f;
  interpolateCoeffs[2] = 0x1.000002p+0f;
  interpolateInput[0] = 0x1.000006p+0f;
  failed |= checkInterpolate(interpolateInput, interpolateCoeffs, "contract split");
  return failed;
}

int main(void) {
  int failed = checkContractSplit();
  failed |= checkNonFiniteTapZero("infinite tap zero", INFINITY);
  failed |= checkNonFiniteTapZero("NaN tap zero", NAN);
  failed |= checkFiniteSignedZeroSkip();

  float decimateInput[kDecimateInput];
  float decimateCoeffs[kDecimateTaps];
  float interpolateInput[kInterpolateInput];
  float interpolateCoeffs[kInterpolateTaps];
  uint32_t state = UINT32_C(0x9E3779B9);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    snprintf(label, sizeof label, "trial %d", trial);
    for (int i = 0; i < kDecimateInput; ++i)
      decimateInput[i] = randomValue(&state);
    for (int i = 0; i < kDecimateTaps; ++i)
      decimateCoeffs[i] = randomValue(&state);
    for (int i = 0; i < kInterpolateInput; ++i)
      interpolateInput[i] = randomValue(&state);
    for (int i = 0; i < kInterpolateTaps; ++i)
      interpolateCoeffs[i] = randomValue(&state);
    failed |= checkDecimate(decimateInput, decimateCoeffs, label);
    failed |= checkInterpolate(interpolateInput, interpolateCoeffs, label);
  }
  return failed;
}
