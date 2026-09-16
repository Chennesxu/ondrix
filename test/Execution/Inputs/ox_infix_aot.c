#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { kLength = 32 };

static int16_t saturate16(int32_t value) {
  return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
}

// nearest_ties_positive at the Q1.15 product boundary: add half, floor.
static int16_t mult(int16_t lhs, int16_t rhs) {
  return saturate16(((int32_t)lhs * rhs + (1 << 14)) >> 15);
}

static int16_t half(int16_t value) { return saturate16(((int32_t)value + 1) >> 1); }

static int16_t add(int16_t lhs, int16_t rhs) { return saturate16((int32_t)lhs + rhs); }

static int16_t sub(int16_t lhs, int16_t rhs) { return saturate16((int32_t)lhs - rhs); }

// The source tree x + y * z - shift(x, amount=-1), and the two trees the same
// tokens would denote under a different grouping or an association through
// the saturation boundary; the corpus must tell all three apart.
static int16_t written(int16_t x, int16_t y, int16_t z) { return sub(add(x, mult(y, z)), half(x)); }
static int16_t leftGrouped(int16_t x, int16_t y, int16_t z) {
  return sub(mult(add(x, y), z), half(x));
}
static int16_t reassociated(int16_t x, int16_t y, int16_t z) {
  return add(x, sub(mult(y, z), half(x)));
}

static int16_t sample(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return (int16_t)((int32_t)(*state % 65536u) - 32768);
}

int main(void) {
  int16_t x[kLength], y[kLength], z[kLength], infix[kLength], calls[kLength];
  uint32_t state = 0x2F6E19A3u;
  for (int i = 0; i < kLength; ++i) {
    x[i] = sample(&state);
    y[i] = sample(&state);
    z[i] = sample(&state);
  }
  // The rails: x + y*z saturates, and (x + y) saturates before the product.
  x[0] = 32767;
  y[0] = 32767;
  z[0] = 32767;
  x[1] = -32768;
  y[1] = 32767;
  z[1] = -32768;
  x[2] = 20000;
  y[2] = 20000;
  z[2] = 16384;

  ondrix_q15_infix_precedence(x, y, z, infix);
  ondrix_q15_infix_calls(x, y, z, calls);
  int groupingSeen = 0, associationSeen = 0;
  for (int i = 0; i < kLength; ++i) {
    int16_t expected = written(x[i], y[i], z[i]);
    if (infix[i] != expected || calls[i] != expected) {
      fprintf(stderr, "[%d]: infix %d, calls %d, expected %d\n", i, infix[i], calls[i], expected);
      return 1;
    }
    groupingSeen |= leftGrouped(x[i], y[i], z[i]) != expected;
    associationSeen |= reassociated(x[i], y[i], z[i]) != expected;
  }
  if (!groupingSeen || !associationSeen) {
    fprintf(stderr, "the corpus does not separate the written tree from its alternatives\n");
    return 1;
  }
  return 0;
}
