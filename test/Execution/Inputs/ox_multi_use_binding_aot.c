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

extern void _mlir_ciface_q15_multi_use_binding(MemRefI16 *, MemRefI16 *, MemRefI16 *);

enum { kLength = 32, kTrialCount = 8 };

/* Independent reference for `add(shift(mult(t, t), -1), t)` with `t =
 * mult(x, y)`: exact integer products, then the one declared boundary per
 * operation, written per rule rather than shared with the requantizer. */
static int16_t narrowSaturating(int32_t value) {
  return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
}

static int16_t roundShiftHalfEven(int32_t value, int shift) {
  int32_t quotient = value >> shift;
  int32_t remainder = value - (quotient << shift);
  int32_t half = (int32_t)1 << (shift - 1);
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return narrowSaturating(quotient);
}

static int16_t multQ15(int16_t lhs, int16_t rhs) {
  return roundShiftHalfEven((int32_t)lhs * (int32_t)rhs, 15);
}

/* `saturations` and `ties` make the corpus claim non-vacuous: without them a
 * corpus that never leaves the linear region would gate nothing about the
 * declared boundaries. */
static int16_t reference(int16_t x, int16_t y, int64_t *saturations, int64_t *ties) {
  int16_t t = multQ15(x, y);
  if ((int32_t)x * (int32_t)y >= ((int32_t)1 << 30))
    ++*saturations;
  int16_t square = multQ15(t, t);
  if (square & 1)
    ++*ties;
  int16_t scaled = roundShiftHalfEven(square, 1);
  return narrowSaturating((int32_t)scaled + (int32_t)t);
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
  int failed = 0;
  int64_t saturations = 0;
  int64_t ties = 0;
  uint32_t state = 0x2F6E19A3u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t x[kLength];
    int16_t y[kLength];
    for (int64_t i = 0; i < kLength; ++i) {
      state = nextState(state);
      x[i] = toSigned16(state);
      state = nextState(state);
      y[i] = toSigned16(state);
    }
    if (trial == 0) {
      /* The one input pair whose exact product is +1.0 and therefore leaves
       * Q1.15 at the first boundary, plus the odd squares that separate the
       * tie rules at the shift. */
      for (int64_t i = 0; i < kLength; ++i) {
        x[i] = INT16_MIN;
        y[i] = (int16_t)(i % 2 == 0 ? INT16_MIN : -3);
      }
    }

    MemRefI16 xRef = {x, x, 0, {kLength}, {1}};
    MemRefI16 yRef = {y, y, 0, {kLength}, {1}};
    MemRefI16 got;
    _mlir_ciface_q15_multi_use_binding(&got, &xRef, &yRef);
    for (int64_t i = 0; i < kLength; ++i) {
      int16_t expected = reference(x[i], y[i], &saturations, &ties);
      int16_t actual = got.aligned[got.offset + i];
      if (actual != expected) {
        fprintf(stderr, "trial %d element %lld: got %d, expected %d\n", trial, (long long)i, actual,
                expected);
        failed = 1;
      }
    }
    free(got.allocated);
  }
  if (saturations == 0 || ties == 0) {
    fprintf(stderr, "corpus is vacuous: %lld product saturations, %lld shift ties\n",
            (long long)saturations, (long long)ties);
    failed = 1;
  }
  return failed;
}
