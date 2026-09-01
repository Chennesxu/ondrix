/* Independent Q31 LMS reference.
 *
 * It reproduces every declared boundary in order -- the per-product tap
 * requantization, the output export, the saturating error, the step, and each
 * weight update -- because the quantized weights feed the next sample and a
 * one-LSB difference in any of them compounds. The product shift is derived
 * from the tap count here rather than taken as a constant.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

/* Two results arrive as one aggregate, not as two output pointers. */
typedef struct {
  MemRefI32 error;
  MemRefI32 adapted;
} LmsResult;

typedef void (*LmsFn)(LmsResult *, MemRefI32 *, MemRefI32 *, MemRefI32 *);

extern void _mlir_ciface_lms_k16_q31(LmsResult *, MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_lms_k5_q31_floor(LmsResult *, MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_lms_k1_q31(LmsResult *, MemRefI32 *, MemRefI32 *, MemRefI32 *);

enum { kEven = 0, kFloor = 1, kSamples = 48 };

/* floor(log2 taps), the same smallest-shift derivation matmul states: ceil
 * would spend one bit of precision at every non-power-of-two tap count. */
static unsigned productShift(int64_t taps) {
  unsigned bits = 0;
  while (((int64_t)1 << (bits + 1)) <= taps)
    ++bits;
  unsigned exact = 62u + bits;
  return exact <= 62u ? 0u : exact - 62u;
}

static int64_t roundShift(int64_t value, unsigned shift, int mode) {
  if (shift == 0)
    return value;
  int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  if (mode == kFloor)
    return quotient;
  int64_t half = divisor >> 1;
  if (remainder > half)
    return quotient + 1;
  if (remainder < half)
    return quotient;
  return (quotient & 1) ? quotient + 1 : quotient;
}

static int32_t saturate(int64_t value) {
  if (value > INT32_MAX)
    return INT32_MAX;
  if (value < INT32_MIN)
    return INT32_MIN;
  return (int32_t)value;
}

static void reference(const int32_t *x, const int32_t *d, const int32_t *w0, int64_t taps,
                      int64_t mu, int productMode, int32_t *error, int32_t *weights) {
  unsigned shift = productShift(taps);
  for (int64_t k = 0; k < taps; ++k)
    weights[k] = w0[k];
  for (int64_t n = 0; n < kSamples; ++n) {
    int64_t acc = 0;
    for (int64_t k = 0; k < taps; ++k) {
      int64_t sample = (n - k) >= 0 ? (int64_t)x[n - k] : 0;
      acc += roundShift((int64_t)weights[k] * sample, shift, productMode);
    }
    int32_t y = saturate(roundShift(acc, 31 - shift, kEven));
    error[n] = saturate((int64_t)d[n] - (int64_t)y);
    int32_t step = saturate(roundShift(mu * (int64_t)error[n], 31, kEven));
    for (int64_t k = 0; k < taps; ++k) {
      int64_t sample = (n - k) >= 0 ? (int64_t)x[n - k] : 0;
      int64_t delta = roundShift((int64_t)step * sample, 31, kEven);
      weights[k] = saturate((int64_t)weights[k] + delta);
    }
  }
}

static int failures = 0;
static uint32_t state = 0x13579bdfu;

static int32_t nextRandom(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (int32_t)state;
}

static void check(const char *name, LmsFn kernel, const int32_t *x, const int32_t *d,
                  const int32_t *w0, int64_t taps, int productMode) {
  MemRefI32 xin = {(int32_t *)x, (int32_t *)x, 0, {kSamples}, {1}};
  MemRefI32 din = {(int32_t *)d, (int32_t *)d, 0, {kSamples}, {1}};
  MemRefI32 win = {(int32_t *)w0, (int32_t *)w0, 0, {taps}, {1}};
  LmsResult out;
  kernel(&out, &xin, &din, &win);
  static int32_t expectedError[kSamples], expectedWeights[64];
  reference(x, d, w0, taps, 134217728, productMode, expectedError, expectedWeights);
  for (int64_t n = 0; n < kSamples; ++n) {
    int32_t observed = out.error.aligned[out.error.offset + n * out.error.strides[0]];
    if (observed != expectedError[n]) {
      printf("%s error[%lld]: observed %d expected %d\n", name, (long long)n, observed,
             expectedError[n]);
      ++failures;
    }
  }
  for (int64_t k = 0; k < taps; ++k) {
    int32_t observed = out.adapted.aligned[out.adapted.offset + k * out.adapted.strides[0]];
    if (observed != expectedWeights[k]) {
      printf("%s weight[%lld]: observed %d expected %d\n", name, (long long)k, observed,
             expectedWeights[k]);
      ++failures;
    }
  }
  free(out.error.allocated);
  free(out.adapted.allocated);
}

int main(void) {
  static int32_t x[kSamples], d[kSamples], w[64];

  const int32_t rails[] = {0, INT32_MAX, INT32_MIN, 1 << 28, -(1 << 28)};
  for (unsigned rail = 0; rail < sizeof(rails) / sizeof(rails[0]); ++rail) {
    for (int64_t n = 0; n < kSamples; ++n) {
      x[n] = rails[rail];
      d[n] = rails[rail];
    }
    for (int64_t k = 0; k < 64; ++k)
      w[k] = rails[rail];
    check("k16_rail", _mlir_ciface_lms_k16_q31, x, d, w, 16, kEven);
    check("k5_rail", _mlir_ciface_lms_k5_q31_floor, x, d, w, 5, kFloor);
    check("k1_rail", _mlir_ciface_lms_k1_q31, x, d, w, 1, kEven);
  }

  /* A constructed double-rounding witness for the non-power-of-two tap count.
   * At n = 1 the two products are 2^30 and 4: the contract's p=2 accumulates
   * 2^28 + 1 and exports 1, while a ceil-derived p=3 accumulates exactly 2^27
   * and its nearest-even export ties to 0. A random corpus does not reach this
   * — an oracle deriving the shift the other way passes every trial below. */
  for (int64_t n = 0; n < kSamples; ++n) {
    x[n] = 0;
    d[n] = 0;
  }
  x[0] = 1;
  x[1] = 1;
  for (int64_t k = 0; k < 64; ++k)
    w[k] = 0;
  w[0] = 1 << 30;
  w[1] = 4;
  check("k5_double_rounding", _mlir_ciface_lms_k5_q31_floor, x, d, w, 5, kFloor);

  for (int trial = 0; trial < 120; ++trial) {
    for (int64_t n = 0; n < kSamples; ++n) {
      x[n] = nextRandom();
      d[n] = nextRandom();
    }
    for (int64_t k = 0; k < 64; ++k)
      w[k] = nextRandom() >> 4;
    check("k16", _mlir_ciface_lms_k16_q31, x, d, w, 16, kEven);
    check("k5", _mlir_ciface_lms_k5_q31_floor, x, d, w, 5, kFloor);
    check("k1", _mlir_ciface_lms_k1_q31, x, d, w, 1, kEven);
  }

  if (failures) {
    printf("lms_q31: %d failures\n", failures);
    return 1;
  }
  return 0;
}
