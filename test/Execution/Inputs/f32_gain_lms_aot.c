#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the f32 gain and lms contracts. Both are exact, so every
 * comparison is bit for bit against a reference that computes the declared
 * event graph itself. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

typedef struct {
  MemRefF32Rank1 error;
  MemRefF32Rank1 adapted;
} LmsResult;

extern void _mlir_ciface_f32_gain_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_gain_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_gain_fast(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_lms_off(LmsResult *, MemRefF32Rank1 *, MemRefF32Rank1 *,
                                     MemRefF32Rank1 *);
extern void _mlir_ciface_f32_lms_fma(LmsResult *, MemRefF32Rank1 *, MemRefF32Rank1 *,
                                     MemRefF32Rank1 *);
extern void _mlir_ciface_f32_lms_fast(LmsResult *, MemRefF32Rank1 *, MemRefF32Rank1 *,
                                      MemRefF32Rank1 *);
extern void _mlir_ciface_f32_lms(LmsResult *, MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);

enum { kGainLength = 16, kSamples = 32, kTaps = 4, kTrialCount = 24 };

static const float kGain = 3.750000e-01f;
static const float kStepSize = 6.250000e-02f;

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int compare(const char *label, const char *mode, int64_t index, float got, float expected) {
  if (floatBits(got) == floatBits(expected))
    return 0;
  fprintf(stderr, "%s %s [%lld]: got %a, expected %a\n", label, mode, (long long)index, (double)got,
          (double)expected);
  return 1;
}

/* One multiply per element under every declaration. */
static int checkGain(const float *input, const char *label) {
  float copy[kGainLength];
  memcpy(copy, input, sizeof(copy));
  MemRefF32Rank1 inputRef = {copy, copy, 0, {kGainLength}, {1}};
  MemRefF32Rank1 off, fma, fast;
  _mlir_ciface_f32_gain_off(&off, &inputRef);
  _mlir_ciface_f32_gain_fma(&fma, &inputRef);
  _mlir_ciface_f32_gain_fast(&fast, &inputRef);

  int failed = 0;
  for (int64_t i = 0; i < kGainLength; ++i) {
    const float expected = input[i] * kGain;
    failed |= compare(label, "gain off", i, off.aligned[off.offset + i * off.strides[0]], expected);
    failed |= compare(label, "gain fma", i, fma.aligned[fma.offset + i * fma.strides[0]], expected);
    /* fast must agree bitwise: see the operation description. */
    failed |=
        compare(label, "gain fast", i, fast.aligned[fast.offset + i * fast.strides[0]], expected);
  }
  free(off.allocated);
  free(fma.allocated);
  free(fast.allocated);
  return failed;
}

/* Per sample: the contract-indexed tap reduction from +0.0, one subtract for
 * the error, one multiply for the step, and one contract-indexed update per
 * weight. The prehistory samples are evaluated, not skipped. */
static void referenceLms(const float *input, const float *desired, const float *weights, int fused,
                         float *error, float *adapted) {
  float state[kTaps];
  memcpy(state, weights, sizeof(state));
  for (int64_t n = 0; n < kSamples; ++n) {
    float output = 0.0f;
    for (int64_t k = 0; k < kTaps; ++k) {
      const float sample = n >= k ? input[n - k] : 0.0f;
      output = fused ? fmaf(state[k], sample, output) : output + state[k] * sample;
    }
    error[n] = desired[n] - output;
    const float step = kStepSize * error[n];
    for (int64_t k = 0; k < kTaps; ++k) {
      const float sample = n >= k ? input[n - k] : 0.0f;
      state[k] = fused ? fmaf(step, sample, state[k]) : state[k] + step * sample;
    }
  }
  memcpy(adapted, state, sizeof(state));
}

static int checkLmsResult(const LmsResult *result, const float *expectedError,
                          const float *expectedAdapted, const char *label, const char *mode) {
  int failed = 0;
  for (int64_t n = 0; n < kSamples; ++n)
    failed |= compare(label, mode, n,
                      result->error.aligned[result->error.offset + n * result->error.strides[0]],
                      expectedError[n]);
  for (int64_t k = 0; k < kTaps; ++k)
    failed |=
        compare(label, mode, k,
                result->adapted.aligned[result->adapted.offset + k * result->adapted.strides[0]],
                expectedAdapted[k]);
  return failed;
}

static int checkLms(const float *input, const float *desired, const float *weights,
                    const char *label) {
  float inputCopy[kSamples];
  float desiredCopy[kSamples];
  float weightCopy[kTaps];
  memcpy(inputCopy, input, sizeof(inputCopy));
  memcpy(desiredCopy, desired, sizeof(desiredCopy));
  memcpy(weightCopy, weights, sizeof(weightCopy));

  MemRefF32Rank1 inputRef = {inputCopy, inputCopy, 0, {kSamples}, {1}};
  MemRefF32Rank1 desiredRef = {desiredCopy, desiredCopy, 0, {kSamples}, {1}};
  MemRefF32Rank1 weightRef = {weightCopy, weightCopy, 0, {kTaps}, {1}};
  LmsResult off, fma, fast, source;
  _mlir_ciface_f32_lms_off(&off, &inputRef, &desiredRef, &weightRef);
  _mlir_ciface_f32_lms_fma(&fma, &inputRef, &desiredRef, &weightRef);
  _mlir_ciface_f32_lms_fast(&fast, &inputRef, &desiredRef, &weightRef);
  _mlir_ciface_f32_lms(&source, &inputRef, &desiredRef, &weightRef);

  float expectedError[kSamples];
  float expectedAdapted[kTaps];
  int failed = 0;
  referenceLms(input, desired, weights, 0, expectedError, expectedAdapted);
  failed |= checkLmsResult(&off, expectedError, expectedAdapted, label, "lms off");
  referenceLms(input, desired, weights, 1, expectedError, expectedAdapted);
  failed |= checkLmsResult(&fma, expectedError, expectedAdapted, label, "lms fma");
  /* fast spends F here: the fused chain over a rounded product and an add. */
  failed |= checkLmsResult(&fast, expectedError, expectedAdapted, label, "lms fast");
  /* The .ox binding declares fma. */
  failed |= checkLmsResult(&source, expectedError, expectedAdapted, label, "lms .ox");

  /* The initial weights are the caller's, and the recursion must not adapt
   * them in place. */
  if (memcmp(weightCopy, weights, sizeof(weightCopy)) != 0) {
    fprintf(stderr, "%s: lms adapted the caller's initial weights in place\n", label);
    failed = 1;
  }
  free(off.error.allocated);
  free(off.adapted.allocated);
  free(fast.error.allocated);
  free(fast.adapted.allocated);
  free(fma.error.allocated);
  free(fma.adapted.allocated);
  free(source.error.allocated);
  free(source.adapted.allocated);
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

/* A directed split witness for lms; gain has no fused form to split. */
static int checkContractSplit(void) {
  float input[kSamples] = {0};
  float desired[kSamples] = {0};
  float weights[kTaps] = {0};

  /* Full-mantissa samples, targets and initial weights make the tap products
   * and the weight updates inexact, so a fused term and a separately rounded
   * term diverge in the exported error signal. */
  const float samples[4] = {0x1.3c6ef3p+0f, -0x1.1e2d5bp+0f, 0x1.7a4c9dp+0f, 0x1.05b2c7p+0f};
  const float targets[3] = {0x1.9e3779p+0f, -0x1.4a7f2bp+0f, 0x1.62e43p+0f};
  const float initial[kTaps] = {0x1.000002p+0f, -0x1.7ffffep-1f, 0x1.3bd3ccp-2f, -0x1.0f876cp-3f};
  for (int i = 0; i < 4; ++i)
    input[i] = samples[i];
  for (int i = 0; i < 3; ++i)
    desired[i] = targets[i];
  memcpy(weights, initial, sizeof weights);
  return checkLms(input, desired, weights, "contract split");
}

int main(void) {
  int failed = checkContractSplit();

  float gainInput[kGainLength];
  float input[kSamples];
  float desired[kSamples];
  float weights[kTaps];
  uint32_t state = UINT32_C(0x1F123BB5);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    snprintf(label, sizeof label, "trial %d", trial);
    for (int i = 0; i < kGainLength; ++i)
      gainInput[i] = randomValue(&state);
    for (int i = 0; i < kSamples; ++i) {
      input[i] = randomValue(&state);
      desired[i] = randomValue(&state);
    }
    for (int i = 0; i < kTaps; ++i)
      weights[i] = randomValue(&state);
    failed |= checkGain(gainInput, label);
    failed |= checkLms(input, desired, weights, label);
  }
  return failed;
}
