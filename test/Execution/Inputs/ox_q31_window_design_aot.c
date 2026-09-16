// The Q31 window design in the .ox coefficient slot, from source through the
// default pipeline to executed bits, against a reference that derives the
// window from its real-valued definition and walks the ordered saturating
// i64 accumulation the stage declares.
#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_q31_window_spectrum(MemRefI32 *, MemRefI32 *);

enum { kTaps = 9, kInput = 40, kOutput = 32, kTrialCount = 6 };

static int64_t clampI64(__int128 value) {
  if (value > (__int128)INT64_MAX)
    return INT64_MAX;
  if (value < (__int128)INT64_MIN)
    return INT64_MIN;
  return (int64_t)value;
}

/* The symmetric Hamming window from its real-valued definition, quantized
 * once to signed Q1.31 with round-to-nearest and declared saturation. */
static void hammingTable(int32_t taps[kTaps]) {
  const double kTwoPi = 6.28318530717958647692528676655900577;
  for (int64_t n = 0; n < kTaps; ++n) {
    double real = 0.54 - 0.46 * cos(kTwoPi * (double)n / (double)(kTaps - 1));
    double scaled = real * 2147483648.0;
    double lower = floor(scaled);
    int64_t quantized = (int64_t)lower + (scaled - lower > 0.5 ? 1 : 0);
    taps[n] = (int32_t)(quantized > INT32_MAX ? INT32_MAX : quantized);
  }
}

/* Nine Q31 products exceed i64, so each update clamps in tap order; then one
 * nearest-even saturating export and the declared saturating absolute value. */
static int32_t reference(const int32_t *input, const int32_t taps[kTaps], int64_t index,
                         int64_t *clampedUpdates, int64_t *exportRails, int64_t *negatives) {
  int64_t sum = 0;
  for (int64_t k = 0; k < kTaps; ++k) {
    __int128 exact = (__int128)sum + (__int128)input[index + k] * (__int128)taps[k];
    sum = clampI64(exact);
    *clampedUpdates += exact != (__int128)sum;
  }
  int64_t quotient = sum >> 31;
  int64_t remainder = sum - (quotient << 31);
  int64_t half = (int64_t)1 << 30;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient > INT32_MAX || quotient < INT32_MIN)
    ++*exportRails;
  int32_t exported =
      (int32_t)(quotient > INT32_MAX ? INT32_MAX : (quotient < INT32_MIN ? INT32_MIN : quotient));
  if (exported < 0)
    ++*negatives;
  int64_t magnitude = exported < 0 ? -(int64_t)exported : (int64_t)exported;
  return (int32_t)(magnitude > INT32_MAX ? INT32_MAX : magnitude);
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

int main(void) {
  int32_t taps[kTaps];
  hammingTable(taps);
  /* The +1.0 window center is the one coefficient the contract saturates. */
  if (taps[0] != 171798692 || taps[kTaps / 2] != INT32_MAX) {
    fprintf(stderr, "window table is not the declared design: %d ... %d\n", taps[0],
            taps[kTaps / 2]);
    return 1;
  }

  int failed = 0;
  int64_t clampedUpdates = 0;
  int64_t exportRails = 0;
  int64_t negatives = 0;
  uint32_t state = 0x4D9A3E17u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int32_t input[kInput];
    for (int64_t i = 0; i < kInput; ++i) {
      state = nextState(state);
      input[i] = (int32_t)state;
    }
    if (trial == 0)
      for (int64_t i = 0; i < kInput; ++i)
        input[i] = i % 2 == 0 ? INT32_MIN : INT32_MAX;
    if (trial == 1)
      for (int64_t i = 0; i < kInput; ++i)
        input[i] = INT32_MIN;

    MemRefI32 inputRef = {input, input, 0, {kInput}, {1}};
    int32_t output[kOutput];
    MemRefI32 got = {output, output, 0, {kOutput}, {1}};
    _mlir_ciface_q31_window_spectrum(&inputRef, &got);
    for (int64_t i = 0; i < kOutput; ++i) {
      int32_t expected = reference(input, taps, i, &clampedUpdates, &exportRails, &negatives);
      int32_t actual = output[i];
      if (actual != expected) {
        fprintf(stderr, "trial %d output %lld: got %d, expected %d\n", trial, (long long)i, actual,
                expected);
        failed = 1;
      }
    }
  }
  if (clampedUpdates == 0 || exportRails == 0 || negatives == 0) {
    fprintf(stderr, "corpus is vacuous: %lld clamped updates, %lld export rails, %lld negatives\n",
            (long long)clampedUpdates, (long long)exportRails, (long long)negatives);
    failed = 1;
  }
  return failed;
}
