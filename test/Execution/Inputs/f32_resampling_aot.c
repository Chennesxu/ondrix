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
 * is the declared event graph, not an optimization the contract derives. */
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
  MemRefF32Rank1 off, fma, source;
  _mlir_ciface_f32_interpolate_off(&off, &inputRef, &coeffRef);
  _mlir_ciface_f32_interpolate_fma(&fma, &inputRef, &coeffRef);
  _mlir_ciface_f32_fir_interpolate(&source, &inputRef, &coeffRef);

  int failed = 0;
  for (int64_t m = 0; m < kInterpolateOutput; ++m) {
    const float expectedOff = referenceInterpolate(input, coeffs, m, 0);
    const float expectedFma = referenceInterpolate(input, coeffs, m, 1);
    failed |= compare(label, "interpolate off", m, off.aligned[off.offset + m * off.strides[0]],
                      expectedOff);
    failed |= compare(label, "interpolate fma", m, fma.aligned[fma.offset + m * fma.strides[0]],
                      expectedFma);
    /* The .ox binding declares off. */
    failed |= compare(label, "interpolate .ox", m,
                      source.aligned[source.offset + m * source.strides[0]], expectedOff);
  }
  free(off.allocated);
  free(fma.allocated);
  free(source.allocated);
  return failed;
}

/* Skipping the inserted-zero terms and materializing them agree on every
 * finite input and part company at infinity, because 0.0 * inf is NaN. With
 * an infinite tap zero, output 1 skips tap 0 (its upsampled index is odd) and
 * stays finite; a lowering that summed the inserted zeros would return NaN
 * there. */
static int checkInsertedZeroSkip(void) {
  float input[kInterpolateInput] = {0.25f, -0.5f, 0.75f, 1.0f};
  float coeffs[kInterpolateTaps] = {INFINITY, 0.5f, -0.25f};

  MemRefF32Rank1 inputRef = {input, input, 0, {kInterpolateInput}, {1}};
  MemRefF32Rank1 coeffRef = {coeffs, coeffs, 0, {kInterpolateTaps}, {1}};
  MemRefF32Rank1 off;
  _mlir_ciface_f32_interpolate_off(&off, &inputRef, &coeffRef);

  int failed = 0;
  const float skipped = off.aligned[off.offset + off.strides[0]];
  if (!isfinite(skipped)) {
    fprintf(stderr, "inserted-zero skip: output 1 is %a, expected a finite value\n",
            (double)skipped);
    failed = 1;
  }
  failed |= compare("inserted-zero skip", "interpolate off", 1, skipped,
                    referenceInterpolate(input, coeffs, 1, 0));
  free(off.allocated);
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
  float decimateInput[kDecimateInput] = {0};
  float decimateCoeffs[kDecimateTaps] = {0};
  float interpolateInput[kInterpolateInput] = {0};
  float interpolateCoeffs[kInterpolateTaps] = {0};

  decimateInput[0] = 1.0f;
  decimateCoeffs[0] = 0x1.000002p+0f;
  decimateInput[1] = -1.0f;
  decimateCoeffs[1] = 1.0f;
  int failed = checkDecimate(decimateInput, decimateCoeffs, "contract split");

  interpolateInput[0] = 1.0f;
  interpolateCoeffs[0] = 0x1.000002p+0f;
  interpolateInput[1] = -1.0f;
  interpolateCoeffs[2] = 1.0f;
  failed |= checkInterpolate(interpolateInput, interpolateCoeffs, "contract split");
  return failed;
}

int main(void) {
  int failed = checkContractSplit();
  failed |= checkInsertedZeroSkip();

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
