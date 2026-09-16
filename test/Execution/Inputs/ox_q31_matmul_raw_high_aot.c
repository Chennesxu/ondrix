#include <stdint.h>
#include <stdio.h>

enum { kRows = 4, kInner = 16, kColumns = 3 };

static int32_t saturate32(__int128 value) {
  return (int32_t)(value > INT32_MAX ? INT32_MAX : (value < INT32_MIN ? INT32_MIN : value));
}

// raw_high: one floor per term at frac 30, the exact sum, one doubling.
static int32_t rawHigh(const int32_t *a, const int32_t *b, int row, int column) {
  __int128 sum = 0;
  for (int k = 0; k < kInner; ++k)
    sum += ((__int128)a[row * kInner + k] * b[k * kColumns + column]) >> 32;
  return saturate32(sum * 2);
}

// The requantized full-product profile the same shape takes without
// product=raw_high: each term rounded nearest-even by p = 4, the sum exported
// nearest-even by 31 - p.
static __int128 roundHalfEven(__int128 value, unsigned shift) {
  __int128 quotient = value >> shift, remainder = value - (quotient << shift);
  __int128 half = (__int128)1 << (shift - 1);
  return quotient + (remainder > half || (remainder == half && (quotient & 1)));
}

static int32_t requantized(const int32_t *a, const int32_t *b, int row, int column) {
  __int128 sum = 0;
  for (int k = 0; k < kInner; ++k)
    sum += roundHalfEven((__int128)a[row * kInner + k] * b[k * kColumns + column], 4);
  return saturate32(roundHalfEven(sum, 27));
}

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

int main(void) {
  int32_t a[kRows * kInner], b[kInner * kColumns], c[kRows * kColumns];
  uint32_t state = 0x2E1B7A55u;
  int contractsDiffer = 0, saturationSeen = 0;
  for (int trial = 0; trial < 16; ++trial) {
    for (int i = 0; i < kRows * kInner; ++i)
      a[i] = (int32_t)next(&state);
    for (int i = 0; i < kInner * kColumns; ++i)
      b[i] = (int32_t)next(&state);
    if (trial == 0) {
      // Full-scale row against full-scale column: the sum of sixteen floors
      // of just below 2^30 passes 2^31 and the doubling saturates.
      for (int k = 0; k < kInner; ++k) {
        a[k] = INT32_MAX;
        b[k * kColumns] = INT32_MAX;
      }
    }
    ondrix_q31_matmul_raw_high(a, b, c);
    for (int row = 0; row < kRows; ++row)
      for (int column = 0; column < kColumns; ++column) {
        int32_t expected = rawHigh(a, b, row, column);
        int32_t got = c[row * kColumns + column];
        if (got != expected) {
          fprintf(stderr, "trial %d [%d][%d]: got %d, expected %d\n", trial, row, column, got,
                  expected);
          return 1;
        }
        contractsDiffer |= expected != requantized(a, b, row, column);
        saturationSeen |= expected == INT32_MAX || expected == INT32_MIN;
      }
  }
  if (!contractsDiffer || !saturationSeen) {
    fprintf(stderr, "positive controls: contracts differ %d, saturation %d\n", contractsDiffer,
            saturationSeen);
    return 1;
  }
  return 0;
}
