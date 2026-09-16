#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { kLength = 64 };
enum Rounding { kTiesPositive, kNearestEven, kTowardZero, kTowardNegative };

static float fromBits(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

// The contract on the exact real: x * 2^frac is exact in binary64, the
// fraction beside its floor is exact wherever it can reach one half, a NaN
// reads as zero, and the result saturates.
static int64_t quantize(float x, int frac, enum Rounding rounding) {
  int64_t rail = (int64_t)1 << (frac);
  if (isnan(x))
    return 0;
  double value = (double)x * ldexp(1.0, frac);
  if (value >= (double)rail)
    return rail - 1;
  if (value <= -(double)rail)
    return -rail;
  double floored = floor(value), remainder = value - floored;
  int64_t quotient = (int64_t)floored;
  switch (rounding) {
  case kTiesPositive:
    quotient += remainder >= 0.5;
    break;
  case kNearestEven:
    quotient += remainder > 0.5 || (remainder == 0.5 && (quotient & 1));
    break;
  case kTowardZero:
    quotient += value < 0 && remainder > 0;
    break;
  case kTowardNegative:
    break;
  }
  return quotient > rail - 1 ? rail - 1 : quotient;
}

static float dequantize(int64_t value, int frac) { return (float)value * ldexpf(1.0f, -frac); }

static int16_t addQ15(int16_t lhs, int16_t rhs) {
  int32_t sum = (int32_t)lhs + rhs;
  return (int16_t)(sum > 32767 ? 32767 : (sum < -32768 ? -32768 : sum));
}

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

static int fail(const char *what, int index, float x, long long got, long long expected) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  fprintf(stderr, "%s[%d]: x %.9g (0x%08x): got %lld, expected %lld\n", what, index, x, bits, got,
          expected);
  return 1;
}

static int checkBlock(const float *x, int *tieRuleSeen, int *railSeen, int *nanSeen) {
  int16_t q15[kLength], even[kLength];
  int32_t q31[kLength], zero[kLength];
  ondrix_f32_quantize_q15(x, q15);
  ondrix_f32_quantize_q15_even(x, even);
  ondrix_f32_quantize_q31(x, q31);
  ondrix_f32_quantize_q31_zero(x, zero);
  for (int i = 0; i < kLength; ++i) {
    int64_t expected = quantize(x[i], 15, kTiesPositive);
    if (q15[i] != expected)
      return fail("q15", i, x[i], q15[i], expected);
    int64_t expectedEven = quantize(x[i], 15, kNearestEven);
    if (even[i] != expectedEven)
      return fail("q15_even", i, x[i], even[i], expectedEven);
    if (q31[i] != quantize(x[i], 31, kTowardNegative))
      return fail("q31", i, x[i], q31[i], quantize(x[i], 31, kTowardNegative));
    if (zero[i] != quantize(x[i], 31, kTowardZero))
      return fail("q31_zero", i, x[i], zero[i], quantize(x[i], 31, kTowardZero));
    *tieRuleSeen |= expected != expectedEven;
    *railSeen |= isfinite(x[i]) && fabsf(x[i]) >= 1.0f;
    *nanSeen |= isnan(x[i]);
  }
  return 0;
}

int main(void) {
  // Exact halves at both widths and both signs, the smallest value above a
  // half, the rails and the last values below them, the odd integer above 2^23
  // that an add-half construction would carry, both zeros, both infinities,
  // three NaN encodings, the subnormal extremes and the binary32 extremes.
  static const uint32_t directedBits[] = {
      0x37800000u, // 2^-16: 0.5 at q15
      0xB7800000u, // -2^-16
      0x38400000u, // 3 * 2^-16: 1.5 at q15
      0xB8400000u, // -3 * 2^-16
      0x37800001u, // just above the q15 half
      0x30000000u, // 2^-31: 0.5 at q31
      0xB0000000u, // -2^-31
      0x30C00000u, // 3 * 2^-31: 1.5 at q31
      0xB0C00000u, // -3 * 2^-31
      0x3F800000u, // 1.0: both rails
      0xBF800000u, // -1.0: exact at both widths
      0x3F7FFFFFu, // 1 - 2^-24: carries onto the q15 rail
      0xBF800001u, // -1 - 2^-23: below the negative rail
      0x3F7FFF00u, // 32767.5 / 32768 at q15
      0xBF7FFF80u, // -32767.75 / 32768
      0x3B800001u, // (2^23 + 1) * 2^-31: an odd integer at q31
      0x3B800002u, // (2^23 + 2) * 2^-31
      0x00000000u, // +0
      0x80000000u, // -0
      0x7F800000u, // +inf
      0xFF800000u, // -inf
      0x7FC00000u, // quiet NaN
      0x7F800001u, // signalling NaN
      0xFFC00000u, // negative NaN
      0x00000001u, // smallest subnormal
      0x80000001u, // its negative
      0x007FFFFFu, // largest subnormal
      0x7F7FFFFFu, // FLT_MAX
      0xFF7FFFFFu, // -FLT_MAX
      0x3F000000u, // 0.5
      0xBF000000u, // -0.5
      0x3E800000u, // 0.25
      0x38800000u, // 2^-14: 2.0 at q15
      0x37000000u, // 2^-17: 0.25 at q15
      0xB7000000u, // -2^-17
      0x4B000000u, // 2^23
      0x501502F9u, // 1e10
  };
  float x[kLength];
  int tieRuleSeen = 0, railSeen = 0, nanSeen = 0;
  for (int i = 0; i < kLength; ++i)
    x[i] = fromBits(directedBits[i % (sizeof(directedBits) / sizeof(directedBits[0]))]);
  if (checkBlock(x, &tieRuleSeen, &railSeen, &nanSeen))
    return 1;
  uint32_t state = 0x9E3779B9u;
  for (int block = 0; block < 4096; ++block) {
    for (int i = 0; i < kLength; ++i) {
      uint32_t bits = next(&state);
      // Half the draws land in (-2, 2), where the grid is dense enough to
      // reach ties and rails; the rest sample the whole encoding.
      if (block & 1)
        bits = (bits & 0x807FFFFFu) | ((0x70u + (bits >> 23 & 0x1F)) << 23);
      x[i] = fromBits(bits);
    }
    if (checkBlock(x, &tieRuleSeen, &railSeen, &nanSeen))
      return 1;
  }
  if (!tieRuleSeen || !railSeen || !nanSeen) {
    fprintf(stderr, "the corpus never separates the tie rules, reaches the rail, or sees a NaN\n");
    return 1;
  }

  int16_t q15[kLength];
  int32_t q31[kLength];
  float dq15[kLength], dq31[kLength], chained[kLength];
  for (int i = 0; i < kLength; ++i) {
    q15[i] = (int16_t)(next(&state) & 0xFFFF);
    q31[i] = (int32_t)next(&state);
  }
  q15[0] = 32767;
  q15[1] = -32768;
  q31[0] = INT32_MAX;
  q31[1] = INT32_MIN;
  q31[2] = 0x00FFFFFF; // 2^24 - 1: the last exact integer in binary32
  q31[3] = 0x01000001; // 2^24 + 1: the first that rounds
  q31[4] = 0x01000003; // a tie the nearest-even conversion resolves upward
  ondrix_q15_dequantize(q15, dq15);
  ondrix_q31_dequantize(q31, dq31);
  int roundingSeen = 0;
  for (int i = 0; i < kLength; ++i) {
    if (dq15[i] != dequantize(q15[i], 15))
      return fail("dequantize_q15", i, dq15[i], q15[i], 0);
    if (dq31[i] != dequantize(q31[i], 31))
      return fail("dequantize_q31", i, dq31[i], q31[i], 0);
    roundingSeen |= (double)dq31[i] != (double)q31[i] * ldexp(1.0, -31);
  }
  if (!roundingSeen) {
    fprintf(stderr, "no q31 value exercised the binary32 rounding\n");
    return 1;
  }
  // dequantize(quantize(x) + y)
  for (int i = 0; i < kLength; ++i)
    x[i] = fromBits(next(&state) & 0xBFFFFFFFu);
  ondrix_f32_quantize_chain(x, q15, chained);
  for (int i = 0; i < kLength; ++i) {
    float expected = dequantize(addQ15((int16_t)quantize(x[i], 15, kTiesPositive), q15[i]), 15);
    if (chained[i] != expected)
      return fail("chain", i, x[i], (long long)chained[i], (long long)expected);
  }
  return 0;
}
