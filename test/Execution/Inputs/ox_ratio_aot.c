#include <stdint.h>
#include <stdio.h>

enum { kLength = 32, kWideLength = 16 };
enum Rounding { kTiesPositive, kNearestEven, kTowardZero, kTowardNegative };

static int64_t saturate(__int128 value, unsigned width) {
  __int128 high = ((__int128)1 << (width - 1)) - 1, low = -((__int128)1 << (width - 1));
  return (int64_t)(value > high ? high : (value < low ? low : value));
}

static int64_t wrap(__int128 value, unsigned width) {
  uint64_t mask = (UINT64_C(1) << width) - 1, bits = (uint64_t)value & mask;
  if (bits & (UINT64_C(1) << (width - 1)))
    bits |= ~mask;
  return (int64_t)bits;
}

// ratio: the exact scaled dividend over the divisor under the declared tie
// rule; a divisor that is not positive gives the dividend's signed rail.
static int64_t ratio(int64_t x, int64_t d, unsigned width, enum Rounding rounding, int wrapping) {
  if (d <= 0)
    return x == 0 ? 0 : saturate(x < 0 ? (__int128)INT64_MIN : (__int128)INT64_MAX, width);
  __int128 scaled = (__int128)x << (width - 1);
  __int128 quotient = scaled / d, remainder = scaled % d;
  if (remainder < 0) {
    --quotient;
    remainder += d;
  }
  switch (rounding) {
  case kTiesPositive:
    quotient += remainder >= d - remainder;
    break;
  case kNearestEven:
    quotient += remainder > d - remainder || (remainder == d - remainder && (quotient & 1));
    break;
  case kTowardZero:
    quotient += scaled < 0 && remainder != 0;
    break;
  case kTowardNegative:
    break;
  }
  return wrapping ? wrap(quotient, width) : saturate(quotient, width);
}

static int64_t mult(int64_t lhs, int64_t rhs, unsigned width) {
  __int128 product = (__int128)lhs * rhs;
  __int128 half = (__int128)1 << (width - 2);
  return saturate((product + half) >> (width - 1), width);
}

static int64_t offset(int64_t value, int64_t bias, unsigned width) {
  return saturate((__int128)value + bias, width);
}

static int64_t absolute(int64_t value, unsigned width) {
  return saturate(value < 0 ? -(__int128)value : value, width);
}

// x / (y * y + 1)
static int16_t proven(int16_t x, int16_t y) {
  return (int16_t)ratio(x, offset(mult(y, y, 16), 1, 16), 16, kTiesPositive, 0);
}

// x / (abs(y) + 1) + ratio(x, shift(abs(y), amount=-3) + 1024, toward_negative, wrap)
static int32_t wide(int32_t x, int32_t y) {
  int64_t first = ratio(x, offset(absolute(y, 32), 1, 32), 32, kTiesPositive, 0);
  int64_t shifted = (absolute(y, 32) + 4) >> 3; // nearest_ties_positive right shift
  int64_t second = ratio(x, offset(shifted, 1024, 32), 32, kTowardNegative, 1);
  return (int32_t)saturate((__int128)first + second, 32);
}

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

int main(void) {
  int16_t x[kLength], y[kLength], out[kLength], sat[kLength];
  int32_t wx[kWideLength], wy[kWideLength], wout[kWideLength];
  uint32_t state = 0x1234ABCDu;
  for (int i = 0; i < kLength; ++i) {
    x[i] = (int16_t)(next(&state) & 0xFFFF);
    y[i] = (int16_t)(next(&state) & 0xFFFF);
  }
  for (int i = 0; i < kWideLength; ++i) {
    wx[i] = (int32_t)next(&state);
    wy[i] = (int32_t)next(&state);
  }
  // Directed: both rails, quotients that saturate (|x| above the divisor),
  // the divisor at one, a zero and negative divisors for the spelled policy,
  // and dividends of both signs at the smallest divisor.
  static const int16_t directedX[] = {32767, -32768, 1, -1, 12345, -12345, 0, 30000, 3, -3};
  static const int16_t directedY[] = {0, 0, 1, -1, 0, 1, 0, -32768, 7, -7};
  for (unsigned i = 0; i < sizeof(directedX) / sizeof(directedX[0]); ++i) {
    x[i] = directedX[i];
    y[i] = directedY[i];
  }
  wx[0] = INT32_MAX;
  wy[0] = 0;
  wx[1] = INT32_MIN;
  wy[1] = INT32_MIN;
  wx[2] = 1;
  wy[2] = -1;

  ondrix_q15_ratio(x, y, out);
  ondrix_q15_ratio_saturate(x, y, sat);
  ondrix_q31_ratio(wx, wy, wout);
  int saturationSeen = 0, nonpositiveSeen = 0, roundingSeen = 0;
  for (int i = 0; i < kLength; ++i) {
    if (out[i] != proven(x[i], y[i])) {
      fprintf(stderr, "ratio[%d]: x %d y %d: got %d, expected %d\n", i, x[i], y[i], out[i],
              proven(x[i], y[i]));
      return 1;
    }
    int64_t expected = ratio(x[i], y[i], 16, kTowardZero, 0);
    if (sat[i] != expected) {
      fprintf(stderr, "ratio_saturate[%d]: x %d y %d: got %d, expected %lld\n", i, x[i], y[i],
              sat[i], (long long)expected);
      return 1;
    }
    saturationSeen |= y[i] > 0 && (expected == 32767 || expected == -32768);
    nonpositiveSeen |= y[i] <= 0 && x[i] != 0;
    roundingSeen |= y[i] > 0 && expected != ratio(x[i], y[i], 16, kTiesPositive, 0);
  }
  for (int i = 0; i < kWideLength; ++i) {
    if (wout[i] != wide(wx[i], wy[i])) {
      fprintf(stderr, "q31_ratio[%d]: x %d y %d: got %d, expected %d\n", i, wx[i], wy[i], wout[i],
              wide(wx[i], wy[i]));
      return 1;
    }
  }
  if (!saturationSeen || !nonpositiveSeen || !roundingSeen) {
    fprintf(stderr, "the corpus never saturates, never divides by a non-positive value, or never "
                    "separates toward_zero from the default\n");
    return 1;
  }
  return 0;
}
