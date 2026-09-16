#include <stdint.h>
#include <stdio.h>

enum { kLength = 32 };

static int16_t saturate16(int64_t value) {
  return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
}

static int32_t saturate32(int64_t value) {
  return (int32_t)(value > INT32_MAX ? INT32_MAX : (value < INT32_MIN ? INT32_MIN : value));
}

// narrow: the value keeps its position, the low sixteen raw bits round away
// under the declared tie rule, and the one carry past the rail is resolved by
// the declared overflow.
static int16_t narrow(int32_t value, int nearestEven, int wrap) {
  int64_t quotient = value >> 16, remainder = value & 0xFFFF;
  if (nearestEven)
    quotient += remainder > 0x8000 || (remainder == 0x8000 && (quotient & 1));
  else
    quotient += remainder >= 0x8000;
  return wrap ? (int16_t)quotient : saturate16(quotient);
}

static int32_t widen(int16_t value) { return (int32_t)value << 16; }

static int32_t multQ31(int32_t lhs, int32_t rhs) {
  int64_t product = (int64_t)lhs * rhs;
  return saturate32((product + ((int64_t)1 << 30)) >> 31);
}

// narrow(widen(x) * y + widen(x / 2), rounding=nearest_even)
static int16_t chain(int16_t x, int32_t y) {
  int16_t half = (int16_t)(((int32_t)x + 1) >> 1);
  int32_t sum = saturate32((int64_t)multQ31(widen(x), y) + widen(half));
  return narrow(sum, 1, 0);
}

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

int main(void) {
  int16_t x[kLength], narrowed[kLength], even[kLength], chained[kLength];
  int32_t y[kLength], widened[kLength];
  uint32_t state = 0x6D2B79F5u;
  for (int i = 0; i < kLength; ++i) {
    x[i] = (int16_t)(next(&state) & 0xFFFF);
    y[i] = (int32_t)next(&state);
  }
  // The exact halves in both signs and both quotient parities, the rail the
  // round-up crosses, the largest value that does not, and both storage rails.
  static const int32_t directed[] = {0x8000,      0x18000,    -0x8000,   -0x18000,
                                     0x7FFF8000,  0x7FFF7FFF, INT32_MAX, INT32_MIN,
                                     -0x7FFF8000, 0x7FFF,     -0x8001,   0};
  for (unsigned i = 0; i < sizeof(directed) / sizeof(directed[0]); ++i)
    y[i] = directed[i];
  x[0] = 32767;
  x[1] = -32768;

  ondrix_q15_widen(x, widened);
  ondrix_q31_narrow(y, narrowed);
  ondrix_q31_narrow_even(y, even);
  ondrix_q15_conversion_chain(x, y, chained);
  int tieRuleSeen = 0, railSeen = 0;
  for (int i = 0; i < kLength; ++i) {
    if (widened[i] != widen(x[i])) {
      fprintf(stderr, "widen[%d]: x %d: got %d, expected %d\n", i, x[i], widened[i], widen(x[i]));
      return 1;
    }
    if (narrowed[i] != narrow(y[i], 0, 0)) {
      fprintf(stderr, "narrow[%d]: y %d: got %d, expected %d\n", i, y[i], narrowed[i],
              narrow(y[i], 0, 0));
      return 1;
    }
    if (even[i] != narrow(y[i], 1, 1)) {
      fprintf(stderr, "narrow_even[%d]: y %d: got %d, expected %d\n", i, y[i], even[i],
              narrow(y[i], 1, 1));
      return 1;
    }
    if (chained[i] != chain(x[i], y[i])) {
      fprintf(stderr, "chain[%d]: x %d y %d: got %d, expected %d\n", i, x[i], y[i], chained[i],
              chain(x[i], y[i]));
      return 1;
    }
    tieRuleSeen |= narrow(y[i], 0, 0) != narrow(y[i], 1, 0);
    railSeen |= narrow(y[i], 0, 0) != narrow(y[i], 0, 1);
  }
  if (!tieRuleSeen || !railSeen) {
    fprintf(stderr, "the corpus never separates the tie rules or the overflow policies\n");
    return 1;
  }
  return 0;
}
