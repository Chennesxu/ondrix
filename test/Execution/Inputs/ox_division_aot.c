#include <stdint.h>
#include <stdio.h>

enum { kLength = 32 };

static int16_t saturate16(int32_t value) {
  return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
}

// Euclidean quotient and remainder, then the declared tie rule over (q, r).
static int16_t divide(int16_t value, int32_t divisor, int nearestEven) {
  int32_t quotient = value / divisor, remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  int32_t complement = divisor - remainder;
  if (nearestEven)
    quotient += remainder > complement || (remainder == complement && (quotient & 1));
  else
    quotient += remainder >= complement;
  return saturate16(quotient);
}

static int16_t mult(int16_t lhs, int16_t rhs) {
  return saturate16(((int32_t)lhs * rhs + (1 << 14)) >> 15);
}

// x / 3 + y * x / 4 - div(x, divisor=6, rounding=nearest_even)
static int16_t written(int16_t x, int16_t y) {
  int16_t sum = saturate16((int32_t)divide(x, 3, 0) + divide(mult(y, x), 4, 0));
  return saturate16((int32_t)sum - divide(x, 6, 1));
}

static int16_t sample(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return (int16_t)((int32_t)(*state % 65536u) - 32768);
}

int main(void) {
  int16_t x[kLength], y[kLength], output[kLength];
  uint32_t state = 0x2F6E19A3u;
  for (int i = 0; i < kLength; ++i) {
    x[i] = sample(&state);
    y[i] = sample(&state);
  }
  // Exact halves at the even divisor, both signs and both quotient parities
  // (3 -> 0.5, 9 -> 1.5, -3 -> -0.5, -9 -> -1.5), the rails, and the tie the
  // power-of-two divisor reaches.
  static const int16_t directed[] = {3, 9, -3, -9, 15, -15, 32767, -32768, 2, -2, 6, -6};
  for (unsigned i = 0; i < sizeof(directed) / sizeof(directed[0]); ++i) {
    x[i] = directed[i];
    y[i] = 0;
  }

  ondrix_q15_division(x, y, output);
  int tieRuleSeen = 0;
  for (int i = 0; i < kLength; ++i) {
    int16_t expected = written(x[i], y[i]);
    if (output[i] != expected) {
      fprintf(stderr, "[%d]: x %d y %d: got %d, expected %d\n", i, x[i], y[i], output[i], expected);
      return 1;
    }
    tieRuleSeen |= divide(x[i], 6, 1) != divide(x[i], 6, 0);
  }
  if (!tieRuleSeen) {
    fprintf(stderr, "the corpus never reaches a tie at the even divisor\n");
    return 1;
  }
  return 0;
}
