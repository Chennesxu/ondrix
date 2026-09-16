#include <stdint.h>
#include <stdio.h>

enum { kLength = 32 };

static int16_t saturate16(int32_t value) {
  return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
}

// The raw constant is Q1.15, so the product is requantized by 2^15 under the
// declared tie rule: add-half-then-shift, or half-to-even on the floor quotient.
static int16_t gainTiesPositive(int16_t x, int32_t gain) {
  return saturate16(((int32_t)x * gain + (1 << 14)) >> 15);
}

static int16_t gainNearestEven(int16_t x, int32_t gain) {
  int32_t product = (int32_t)x * gain;
  int32_t quotient = product >> 15;
  int32_t remainder = product - (quotient << 15);
  if (remainder > (1 << 14) || (remainder == (1 << 14) && (quotient & 1)))
    ++quotient;
  return saturate16(quotient);
}

// 3 * y + x * 16384 - 1024 - gain(x, gain=16384, rounding=nearest_even)
static int16_t written(int16_t x, int16_t y) {
  int16_t sum = saturate16((int32_t)gainTiesPositive(y, 3) + gainTiesPositive(x, 16384));
  int16_t shifted = saturate16((int32_t)sum - 1024);
  return saturate16((int32_t)shifted - gainNearestEven(x, 16384));
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
  // Odd x is the exact half at gain 16384, with both parities of the
  // quotient and both signs; the rails saturate the offset.
  static const int16_t directed[] = {1, 3, -1, -3, 32767, -32768, 0};
  for (unsigned i = 0; i < sizeof(directed) / sizeof(directed[0]); ++i)
    x[i] = directed[i];

  ondrix_q15_literals(x, y, output);
  int tieRuleSeen = 0;
  for (int i = 0; i < kLength; ++i) {
    int16_t expected = written(x[i], y[i]);
    if (output[i] != expected) {
      fprintf(stderr, "[%d]: x %d y %d: got %d, expected %d\n", i, x[i], y[i], output[i], expected);
      return 1;
    }
    tieRuleSeen |= gainTiesPositive(x[i], 16384) != gainNearestEven(x[i], 16384);
  }
  if (!tieRuleSeen) {
    fprintf(stderr, "the corpus never reaches a tie at the halving gain\n");
    return 1;
  }
  return 0;
}
