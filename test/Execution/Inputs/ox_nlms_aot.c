#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { kSamples = 64, kTaps = 8, kNormalizedMu = 8192, kPlainMu = 4096, kEpsilon = 16 };

static int16_t saturate16(int64_t value) {
  return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
}

// round_half_even(value / 2^shift), saturated to i16 or left in i64.
static int64_t roundHalfEven(int64_t value, unsigned shift) {
  int64_t quotient = value >> shift, remainder = value - (quotient << shift);
  int64_t half = (int64_t)1 << (shift - 1);
  quotient += remainder > half || (remainder == half && (quotient & 1));
  return quotient;
}

// The rounded quotient under nearest-even over the Euclidean pair.
static int16_t roundedQuotient(int64_t dividend, int64_t divisor) {
  int64_t quotient = dividend / divisor, remainder = dividend % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  quotient +=
      remainder > divisor - remainder || (remainder == divisor - remainder && (quotient & 1));
  return saturate16(quotient);
}

// The quantized-state recursion; normalized divides the step by epsilon plus
// the window energy requantized once to Q15.
static void reference(const int16_t *x, const int16_t *d, const int16_t *initial, int normalized,
                      int64_t mu, int16_t *errors, int16_t *weights, int *stepSaturated,
                      int *energySeen) {
  memcpy(weights, initial, kTaps * sizeof(int16_t));
  for (int n = 0; n < kSamples; ++n) {
    int64_t acc = 0, energy = 0;
    for (int k = 0; k < kTaps; ++k) {
      int64_t sample = n - k >= 0 ? x[n - k] : 0;
      acc += (int64_t)weights[k] * sample;
      energy += sample * sample;
    }
    int16_t output = saturate16(roundHalfEven(acc, 15));
    int16_t error = saturate16((int64_t)d[n] - output);
    errors[n] = error;
    int16_t step;
    if (normalized) {
      int64_t divisor = kEpsilon + roundHalfEven(energy, 15);
      int64_t exact = mu * error;
      step = roundedQuotient(exact, divisor);
      *stepSaturated |= (exact >= 0 ? exact : -exact) / divisor > 32767;
      *energySeen |= divisor > kEpsilon;
    } else {
      step = saturate16(roundHalfEven(mu * error, 15));
    }
    for (int k = 0; k < kTaps; ++k) {
      int64_t sample = n - k >= 0 ? x[n - k] : 0;
      int16_t delta = saturate16(roundHalfEven((int64_t)step * sample, 15));
      weights[k] = saturate16((int64_t)weights[k] + delta);
    }
  }
}

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

int main(void) {
  int16_t x[kSamples], d[kSamples], w[kTaps];
  int16_t errors[kSamples], adapted[kTaps], lmsErrors[kSamples], lmsAdapted[kTaps];
  int16_t expectedErrors[kSamples], expectedWeights[kTaps];
  uint32_t state = 0x7C3A9E11u;
  // A loud half, where the energy dominates epsilon, then a quiet half,
  // where the step saturates against epsilon alone.
  for (int n = 0; n < kSamples; ++n) {
    int16_t loud = (int16_t)(next(&state) & 0xFFFF);
    x[n] = n < kSamples / 2 ? loud : (int16_t)(loud / 128);
    d[n] = (int16_t)((next(&state) & 0xFFFF) / 4);
  }
  for (int k = 0; k < kTaps; ++k)
    w[k] = (int16_t)((next(&state) & 0x0FFF) - 0x0800);
  int stepSaturated = 0, energySeen = 0, unused0 = 0, unused1 = 0;

  ondrix_q15_nlms(x, d, w, errors, adapted);
  reference(x, d, w, 1, kNormalizedMu, expectedErrors, expectedWeights, &stepSaturated,
            &energySeen);
  for (int n = 0; n < kSamples; ++n)
    if (errors[n] != expectedErrors[n]) {
      fprintf(stderr, "nlms error[%d]: got %d, expected %d\n", n, errors[n], expectedErrors[n]);
      return 1;
    }
  for (int k = 0; k < kTaps; ++k)
    if (adapted[k] != expectedWeights[k]) {
      fprintf(stderr, "nlms weight[%d]: got %d, expected %d\n", k, adapted[k], expectedWeights[k]);
      return 1;
    }
  // The plain kernel (q15_lms.ox, step 4096) on the same corpus from the same
  // weights: the two recursions must part, or the normalization was not
  // exercised.
  ondrix_q15_lms(x, d, w, lmsErrors, lmsAdapted);
  reference(x, d, w, 0, kPlainMu, expectedErrors, expectedWeights, &unused0, &unused1);
  int differs = memcmp(errors, lmsErrors, sizeof(errors)) != 0;
  if (memcmp(lmsErrors, expectedErrors, sizeof(errors)) != 0 ||
      memcmp(lmsAdapted, expectedWeights, sizeof(adapted)) != 0) {
    fprintf(stderr, "lms control diverged from its reference\n");
    return 1;
  }
  if (!differs || !stepSaturated || !energySeen) {
    fprintf(stderr, "positive controls: differs %d, step saturated %d, energy seen %d\n", differs,
            stepSaturated, energySeen);
    return 1;
  }
  return 0;
}
