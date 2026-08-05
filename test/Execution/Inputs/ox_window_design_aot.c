#include <math.h>
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

extern void _mlir_ciface_q15_window_spectrum(MemRefI16 *, MemRefI16 *);

enum { kTaps = 9, kInput = 40, kOutput = 32, kTrialCount = 6 };

/* The symmetric Hamming window from its real-valued definition, quantized
 * once to signed Q1.15 with round-half-even and declared saturation. Derived
 * here rather than read back from the compiler's table. */
static void hammingTable(int16_t taps[kTaps]) {
  const double kTwoPi = 6.28318530717958647692528676655900577;
  for (int64_t n = 0; n < kTaps; ++n) {
    double real = 0.54 - 0.46 * cos(kTwoPi * (double)n / (double)(kTaps - 1));
    double scaled = real * 32768.0;
    double lower = floor(scaled);
    int64_t quantized = (int64_t)lower + (scaled - lower > 0.5 ? 1 : 0);
    taps[n] = (int16_t)(quantized > 32767 ? 32767 : (quantized < -32768 ? -32768 : quantized));
  }
}

/* Valid-boundary FIR over the exact accumulator (nine Q15 products bound the
 * window sum well inside i35, so no update wraps), one nearest-even
 * saturating export, then the declared saturating absolute value. */
static int16_t reference(const int16_t *input, const int16_t taps[kTaps], int64_t index,
                         int64_t *exportRails, int64_t *negatives) {
  int64_t sum = 0;
  for (int64_t k = 0; k < kTaps; ++k)
    sum += (int64_t)input[index + k] * (int64_t)taps[k];
  int64_t quotient = sum >> 15;
  int64_t remainder = sum - (quotient << 15);
  int64_t half = (int64_t)1 << 14;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient > 32767 || quotient < -32768)
    ++*exportRails;
  int16_t exported = (int16_t)(quotient > 32767 ? 32767 : (quotient < -32768 ? -32768 : quotient));
  if (exported < 0)
    ++*negatives;
  int32_t magnitude = exported < 0 ? -(int32_t)exported : (int32_t)exported;
  return (int16_t)(magnitude > 32767 ? 32767 : magnitude);
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

int main(void) {
  int16_t taps[kTaps];
  hammingTable(taps);
  /* The +1.0 window center is the one coefficient the contract saturates. */
  if (taps[0] != 2621 || taps[kTaps / 2] != 32767) {
    fprintf(stderr, "window table is not the declared design: %d ... %d\n", taps[0],
            taps[kTaps / 2]);
    return 1;
  }

  int failed = 0;
  int64_t exportRails = 0;
  int64_t negatives = 0;
  uint32_t state = 0x4D9A3E17u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t input[kInput];
    for (int64_t i = 0; i < kInput; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    if (trial == 0)
      for (int64_t i = 0; i < kInput; ++i)
        input[i] = (int16_t)(i % 2 == 0 ? INT16_MIN : INT16_MAX);

    MemRefI16 inputRef = {input, input, 0, {kInput}, {1}};
    MemRefI16 got;
    _mlir_ciface_q15_window_spectrum(&got, &inputRef);
    for (int64_t i = 0; i < kOutput; ++i) {
      int16_t expected = reference(input, taps, i, &exportRails, &negatives);
      int16_t actual = got.aligned[got.offset + i];
      if (actual != expected) {
        fprintf(stderr, "trial %d output %lld: got %d, expected %d\n", trial, (long long)i, actual,
                expected);
        failed = 1;
      }
    }
    free(got.allocated);
  }
  if (exportRails == 0 || negatives == 0) {
    fprintf(stderr, "corpus is vacuous: %lld export rails, %lld negative windows\n",
            (long long)exportRails, (long long)negatives);
    failed = 1;
  }
  return failed;
}
