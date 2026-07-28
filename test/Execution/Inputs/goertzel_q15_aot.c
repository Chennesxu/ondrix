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

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI64;

extern void _mlir_ciface_goertzel64_5(MemRefI64 *, MemRefI16 *);
extern void _mlir_ciface_goertzel100_13(MemRefI64 *, MemRefI16 *);
extern void _mlir_ciface_goertzel16_0(MemRefI64 *, MemRefI16 *);

enum { kMaxLength = 100, kTrialCount = 12 };

/* Q15 recursion coefficients q15(cos(2*pi*k/N)) derived independently
 * with 50-digit mpmath: (64, 5) -> 28899, (100, 13) -> 22431,
 * (16, 0) -> 32767 (the saturated +1.0 convention). */

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

static int16_t saturate16(int32_t value) {
  if (value > 32767)
    return 32767;
  if (value < -32768)
    return -32768;
  return (int16_t)value;
}

static int64_t referenceGoertzel(const int16_t *input, int64_t extent, int64_t coefficient) {
  int16_t s1 = 0;
  int16_t s2 = 0;
  for (int64_t n = 0; n < extent; ++n) {
    int16_t m = roundShift15Sat(2 * coefficient * (int64_t)s1);
    int16_t s0 = saturate16((int32_t)input[n] + m - s2);
    s2 = s1;
    s1 = s0;
  }
  int16_t m = roundShift15Sat(2 * coefficient * (int64_t)s1);
  return (int64_t)s1 * s1 + (int64_t)s2 * s2 - (int64_t)m * s2;
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

static int check(void (*kernel)(MemRefI64 *, MemRefI16 *), const int16_t *input, int64_t extent,
                 int64_t coefficient, const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {extent}, {1}};
  MemRefI64 output;
  kernel(&output, &inputRef);

  int64_t expected = referenceGoertzel(input, extent, coefficient);
  int failed = output.sizes[0] != 1;
  if (!failed) {
    int64_t actual = output.aligned[output.offset];
    if (actual != expected) {
      fprintf(stderr, "%s: got %lld, expected %lld\n", label, (long long)actual,
              (long long)expected);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

static int checkAll(const int16_t *input, const char *label) {
  char buffer[48];
  int failed = 0;
  snprintf(buffer, sizeof buffer, "%s (64,5)", label);
  failed |= check(_mlir_ciface_goertzel64_5, input, 64, 28899, buffer);
  snprintf(buffer, sizeof buffer, "%s (100,13)", label);
  failed |= check(_mlir_ciface_goertzel100_13, input, 100, 22431, buffer);
  snprintf(buffer, sizeof buffer, "%s (16,0)", label);
  failed |= check(_mlir_ciface_goertzel16_0, input, 16, 32767, buffer);
  return failed;
}

int main(void) {
  int failed = 0;
  int16_t input[kMaxLength];

  /* In-bin tone for (64, 5): a modest-amplitude cosine at the detected
   * frequency; the (100, 13) and (16, 0) kernels see it as an arbitrary
   * signal, which is equally valid for a bit-exact gate. */
  for (int64_t i = 0; i < kMaxLength; ++i)
    input[i] = (int16_t)(8192.0 * cos(2.0 * 3.14159265358979323846 * 5.0 * (double)i / 64.0));
  failed |= checkAll(input, "tone");

  for (int64_t i = 0; i < kMaxLength; ++i)
    input[i] = 0;
  failed |= checkAll(input, "silence");

  for (int64_t i = 0; i < kMaxLength; ++i)
    input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
  failed |= checkAll(input, "rails");

  for (int64_t i = 0; i < kMaxLength; ++i)
    input[i] = INT16_MIN;
  failed |= checkAll(input, "dc min");

  uint32_t state = 0x60E77E11u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[24];
    for (int64_t i = 0; i < kMaxLength; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "trial %d", trial);
    failed |= checkAll(input, label);
  }
  return failed;
}
