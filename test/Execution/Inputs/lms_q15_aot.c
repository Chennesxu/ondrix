#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

/* Two tensor results come back as one packed pair of descriptors. */
typedef struct {
  MemRefI16 errors;
  MemRefI16 adapted;
} LmsResult;

extern void _mlir_ciface_lms8_q15(LmsResult *, MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_lms1_q15(LmsResult *, MemRefI16 *, MemRefI16 *, MemRefI16 *);

enum { kSamples = 256, kTaps = 8, kMu = 4096, kTrialCount = 8 };

/* Independent per-step contract arithmetic in explicit floor-division
 * form: nearest-even shift by 15 with i16 saturation. */
static int16_t roundShift15Sat(int64_t value) {
  int64_t quotient = value / 32768;
  int64_t remainder = value % 32768;
  if (remainder < 0) {
    --quotient;
    remainder += 32768;
  }
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
    ++quotient;
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return (int16_t)quotient;
}

/* Single nearest-even rounding of the fused triple product by 30 — the
 * real-arithmetic reassociation the contract forbids; used only to pin
 * the compounding witness. */
static int16_t roundShift30Sat(int64_t value) {
  const int64_t divisor = (int64_t)1 << 30;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  const int64_t half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return (int16_t)quotient;
}

static int16_t saturate16(int32_t value) {
  if (value > 32767)
    return 32767;
  if (value < -32768)
    return -32768;
  return (int16_t)value;
}

/* One full reference recursion; fusedUpdate selects the illegal
 * single-rounding update for the witness counters. */
static void referenceLms(const int16_t *x, const int16_t *d, const int16_t *initial,
                         int64_t samples, int64_t taps, int64_t mu, int fusedUpdate,
                         int16_t *errors, int16_t *weights) {
  for (int64_t k = 0; k < taps; ++k)
    weights[k] = initial[k];
  for (int64_t n = 0; n < samples; ++n) {
    int64_t acc = 0;
    for (int64_t k = 0; k < taps; ++k)
      if (n >= k)
        acc += (int64_t)weights[k] * x[n - k];
    int16_t output = roundShift15Sat(acc);
    int16_t error = saturate16((int32_t)d[n] - output);
    errors[n] = error;
    int16_t step = roundShift15Sat((int64_t)mu * error);
    for (int64_t k = 0; k < taps; ++k) {
      int64_t sample = n >= k ? x[n - k] : 0;
      int16_t delta = fusedUpdate ? roundShift30Sat((int64_t)mu * error * sample)
                                  : roundShift15Sat((int64_t)step * sample);
      weights[k] = saturate16((int32_t)weights[k] + delta);
    }
  }
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int16_t toSigned16(uint32_t bits) {
  uint32_t low = bits & 0xFFFFu;
  return (int16_t)(low < 32768u ? (int32_t)low : (int32_t)low - 65536);
}

static int check(void (*kernel)(LmsResult *, MemRefI16 *, MemRefI16 *, MemRefI16 *),
                 const int16_t *x, const int16_t *d, const int16_t *initial, int64_t samples,
                 int64_t taps, int64_t mu, const char *label) {
  MemRefI16 inputRef = {(int16_t *)x, (int16_t *)x, 0, {samples}, {1}};
  MemRefI16 desiredRef = {(int16_t *)d, (int16_t *)d, 0, {samples}, {1}};
  MemRefI16 weightsRef = {(int16_t *)initial, (int16_t *)initial, 0, {taps}, {1}};
  LmsResult result;
  kernel(&result, &inputRef, &desiredRef, &weightsRef);
  MemRefI16 errorsOut = result.errors;
  MemRefI16 adaptedOut = result.adapted;

  int16_t errors[kSamples];
  int16_t weights[kTaps];
  referenceLms(x, d, initial, samples, taps, mu, 0, errors, weights);

  int failed = errorsOut.sizes[0] != samples || adaptedOut.sizes[0] != taps;
  for (int64_t n = 0; n < samples && n < errorsOut.sizes[0]; ++n) {
    int16_t actual = errorsOut.aligned[errorsOut.offset + n * errorsOut.strides[0]];
    if (actual != errors[n]) {
      fprintf(stderr, "%s error %lld: got %d, expected %d\n", label, (long long)n, actual,
              errors[n]);
      failed = 1;
    }
  }
  for (int64_t k = 0; k < taps && k < adaptedOut.sizes[0]; ++k) {
    int16_t actual = adaptedOut.aligned[adaptedOut.offset + k * adaptedOut.strides[0]];
    if (actual != weights[k]) {
      fprintf(stderr, "%s weight %lld: got %d, expected %d\n", label, (long long)k, actual,
              weights[k]);
      failed = 1;
    }
  }
  free(errorsOut.allocated);
  free(adaptedOut.allocated);
  return failed;
}

int main(void) {
  int failed = 0;
  int16_t x[kSamples];
  int16_t d[kSamples];
  int16_t initial[kTaps];

  /* Deterministic corpus for the pinned family-9 compounding witness. */
  uint32_t state = 0x1A751355u;
  for (int64_t i = 0; i < kSamples; ++i) {
    state = nextState(state);
    x[i] = toSigned16(state);
  }
  for (int64_t i = 0; i < kSamples; ++i) {
    state = nextState(state);
    d[i] = toSigned16(state);
  }
  for (int64_t k = 0; k < kTaps; ++k) {
    state = nextState(state);
    initial[k] = toSigned16(state);
  }
  failed |= check(_mlir_ciface_lms8_q15, x, d, initial, kSamples, kTaps, kMu, "witness corpus");

  {
    int16_t contractErrors[kSamples], contractWeights[kTaps];
    int16_t fusedErrors[kSamples], fusedWeights[kTaps];
    referenceLms(x, d, initial, kSamples, kTaps, kMu, 0, contractErrors, contractWeights);
    referenceLms(x, d, initial, kSamples, kTaps, kMu, 1, fusedErrors, fusedWeights);
    int64_t errorDivergences = 0;
    int64_t firstDivergence = -1;
    for (int64_t n = 0; n < kSamples; ++n)
      if (contractErrors[n] != fusedErrors[n]) {
        ++errorDivergences;
        if (firstDivergence < 0)
          firstDivergence = n;
      }
    int64_t weightDivergences = 0;
    for (int64_t k = 0; k < kTaps; ++k)
      weightDivergences += contractWeights[k] != fusedWeights[k];
    if (errorDivergences != 172 || weightDivergences != 6 || firstDivergence != 4) {
      fprintf(stderr,
              "fused-update witness: %lld/%lld errors, %lld weights, first %lld; "
              "pinned 172/256, 6, 4\n",
              (long long)errorDivergences, (long long)kSamples, (long long)weightDivergences,
              (long long)firstDivergence);
      failed = 1;
    }
  }

  /* Directed and random trials for the bit-exact gate. */
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    for (int64_t i = 0; i < kSamples; ++i) {
      state = nextState(state);
      x[i] = toSigned16(state);
    }
    for (int64_t i = 0; i < kSamples; ++i) {
      state = nextState(state);
      d[i] = toSigned16(state);
    }
    for (int64_t k = 0; k < kTaps; ++k) {
      state = nextState(state);
      initial[k] = toSigned16(state);
    }
    if (trial == 0)
      for (int64_t i = 0; i < kSamples; ++i) {
        x[i] = (i & 1) ? INT16_MIN : INT16_MAX;
        d[i] = (i & 2) ? INT16_MAX : INT16_MIN;
      }
    if (trial == 1)
      for (int64_t i = 0; i < kSamples; ++i)
        x[i] = 0; /* no input: weights must never move and e[n] == d[n] */
    snprintf(label, sizeof label, "lms8 trial %d", trial);
    failed |= check(_mlir_ciface_lms8_q15, x, d, initial, kSamples, kTaps, kMu, label);
    snprintf(label, sizeof label, "lms1 trial %d", trial);
    failed |= check(_mlir_ciface_lms1_q15, x, d, initial, 32, 1, 16384, label);
  }
  return failed;
}
