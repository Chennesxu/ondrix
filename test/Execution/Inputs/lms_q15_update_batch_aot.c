#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for vertical batching of the Q15 LMS weight update, and the
 * lane-order witness for it.
 *
 * The batched block loads the sample span forward and reverses the lanes,
 * because index k reads x[n - k]. A reversed or rotated lane order is
 * therefore a live failure mode that no structural check can rule out, so this
 * harness runs asymmetric samples and compares bits against a reference
 * written from the declared contract alone: an exact i64 filter sum, one
 * nearest-even saturating boundary per rounding site, and a saturating i16
 * state add per tap.
 *
 * Non-vacuity is enforced, not assumed. The same reference is run a second
 * time with the batched taps (0..7 of 11, for the steady-state outputs n >= 10)
 * reading their span reversed, and the harness fails unless that mis-ordered
 * schedule actually produces different final weights. A symmetric corpus would
 * pass either lane order and is rejected here. */

enum {
  kSamples = 64,
  kTaps = 11,
  kVectorWidth = 8,
  kTrials = 20,
};

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

typedef struct {
  MemRefI16 error;
  MemRefI16 adapted;
} LmsResult;

extern void _mlir_ciface_lms_q15(LmsResult *, MemRefI16 *, MemRefI16 *, MemRefI16 *);

static const int64_t kStepSize = 4096;

/* floor(x / 2^15) with the tie broken to even, then clamped to i16. */
static int16_t roundShiftQ15(int64_t value) {
  int64_t quotient = value >> 15;
  uint64_t remainder = (uint64_t)value & ((UINT64_C(1) << 15) - 1);
  const uint64_t half = UINT64_C(1) << 14;
  if (remainder > half || (remainder == half && (quotient & 1)))
    quotient += 1;
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return (int16_t)quotient;
}

static int16_t satCastI16(int32_t value) {
  if (value > 32767)
    return 32767;
  if (value < -32768)
    return -32768;
  return (int16_t)value;
}

static int16_t sampleAt(const int16_t *x, int64_t index) {
  return index < 0 ? (int16_t)0 : x[index];
}

/* The declared adaptive step, per output n: an exact i64 filter sum, the
 * rounded output, the saturating error, the rounded step scaling, and one
 * saturating state update per tap. `reverseLanes` mis-orders exactly the taps
 * the batched block covers, which is what makes the corpus discriminating. */
static void referenceLms(const int16_t *x, const int16_t *d, const int16_t *w0, int16_t *error,
                         int16_t *weights, int reverseLanes) {
  memcpy(weights, w0, sizeof(int16_t) * kTaps);
  const int64_t batched = (kTaps / kVectorWidth) * kVectorWidth;
  for (int64_t n = 0; n < kSamples; ++n) {
    int64_t accumulator = 0;
    for (int64_t k = 0; k < kTaps; ++k)
      accumulator += (int64_t)weights[k] * (int64_t)sampleAt(x, n - k);
    int16_t filtered = roundShiftQ15(accumulator);
    int16_t residual = satCastI16((int32_t)d[n] - (int32_t)filtered);
    error[n] = residual;
    int16_t scaledError = roundShiftQ15((int64_t)residual * kStepSize);
    for (int64_t k = 0; k < kTaps; ++k) {
      int64_t tap = k;
      if (reverseLanes && n >= kTaps - 1 && k < batched)
        tap = batched - 1 - k;
      int16_t step = roundShiftQ15((int64_t)scaledError * (int64_t)sampleAt(x, n - tap));
      weights[k] = satCastI16((int32_t)weights[k] + (int32_t)step);
    }
  }
}

static uint32_t nextRandom(uint32_t *state) {
  *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
  return *state >> 8;
}

/* Sixteen pseudorandom trials plus four directed rails: the positive rail,
 * alternating rails, an impulse, and a ramp against the negative rail. */
static void buildTrial(int trial, int16_t *x, int16_t *d, int16_t *w0) {
  uint32_t state = (uint32_t)(trial * 2654435761u + 12345u);
  for (int64_t i = 0; i < kSamples; ++i) {
    x[i] = (int16_t)(nextRandom(&state) & 0xFFFF);
    d[i] = (int16_t)(nextRandom(&state) & 0xFFFF);
  }
  for (int64_t k = 0; k < kTaps; ++k)
    w0[k] = (int16_t)(nextRandom(&state) & 0xFFFF);

  if (trial < 16)
    return;
  for (int64_t i = 0; i < kSamples; ++i) {
    switch (trial) {
    case 16:
      x[i] = 32767;
      d[i] = -32768;
      break;
    case 17:
      x[i] = (i & 1) ? (int16_t)-32768 : (int16_t)32767;
      d[i] = (int16_t)32767;
      break;
    case 18:
      x[i] = i == 13 ? (int16_t)32767 : (int16_t)0;
      d[i] = i == 13 ? (int16_t)-32768 : (int16_t)1;
      break;
    default:
      x[i] = (int16_t)(i * 509 - 32768);
      d[i] = (int16_t)-32768;
      break;
    }
  }
  for (int64_t k = 0; k < kTaps; ++k)
    w0[k] = trial == 18 ? (int16_t)(k * 2971 - 16384) : (int16_t)32767;
}

int main(void) {
  int failures = 0;
  int discriminating = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    int16_t x[kSamples], d[kSamples], w0[kTaps];
    buildTrial(trial, x, d, w0);

    int16_t expectedError[kSamples], expectedWeights[kTaps];
    referenceLms(x, d, w0, expectedError, expectedWeights, 0);
    int16_t misorderedError[kSamples], misorderedWeights[kTaps];
    referenceLms(x, d, w0, misorderedError, misorderedWeights, 1);
    if (memcmp(expectedWeights, misorderedWeights, sizeof(expectedWeights)) != 0)
      ++discriminating;

    MemRefI16 inputRef = {x, x, 0, {kSamples}, {1}};
    MemRefI16 desiredRef = {d, d, 0, {kSamples}, {1}};
    MemRefI16 weightRef = {w0, w0, 0, {kTaps}, {1}};
    LmsResult result;
    _mlir_ciface_lms_q15(&result, &inputRef, &desiredRef, &weightRef);

    for (int64_t n = 0; n < kSamples; ++n) {
      int16_t got = result.error.aligned[result.error.offset + n * result.error.strides[0]];
      if (got != expectedError[n]) {
        fprintf(stderr, "trial %d error[%lld]: got %d, expected %d\n", trial, (long long)n,
                (int)got, (int)expectedError[n]);
        ++failures;
      }
    }
    for (int64_t k = 0; k < kTaps; ++k) {
      int16_t got = result.adapted.aligned[result.adapted.offset + k * result.adapted.strides[0]];
      if (got != expectedWeights[k]) {
        fprintf(stderr, "trial %d weight[%lld]: got %d, expected %d\n", trial, (long long)k,
                (int)got, (int)expectedWeights[k]);
        ++failures;
      }
    }
  }

  if (discriminating == 0) {
    fprintf(stderr, "corpus does not separate the lane order\n");
    return 1;
  }
  printf("lane-order discriminating trials: %d of %d\n", discriminating, kTrials);
  return failures != 0;
}
