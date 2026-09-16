#include <math.h>
#include <stdint.h>
#include <stdio.h>

// The coefficients are quantized here from a long double cosine, sharing no
// code with the compiler's tie-guarded table (the independence the Q31 DCT
// gate already relies on).
static int64_t coefficient(int64_t extent, int64_t k, int64_t n) {
  static const long double kPi = 3.14159265358979323846264338327950288419716939937510582097494459L;
  long double angle = kPi * (long double)((2 * n + 1) * k) / (2.0L * (long double)extent);
  long double scaled = cosl(angle) * 2147483648.0L;
  long double lower = floorl(scaled);
  long double fraction = scaled - lower;
  int64_t quantized = (int64_t)lower;
  if (fraction > 0.5L || (fraction == 0.5L && (quantized & 1)))
    ++quantized;
  return quantized > INT32_MAX ? INT32_MAX : quantized;
}

static int64_t roundShift(__int128 value, unsigned shift, int nearestEven) {
  __int128 quotient = value >> shift, remainder = value - (quotient << shift);
  if (nearestEven) {
    __int128 half = (__int128)1 << (shift - 1);
    quotient += remainder > half || (remainder == half && (quotient & 1));
  }
  return (int64_t)quotient;
}

static int32_t saturate32(int64_t value) {
  return (int32_t)(value > INT32_MAX ? INT32_MAX : (value < INT32_MIN ? INT32_MIN : value));
}

// raw_high: one floor per product at frac 30, the exact row sum, one export
// shift by m onto frac 30 - m under the declared rounding.
static int32_t rawHigh(const int32_t *input, int64_t extent, int64_t k, unsigned m,
                       int nearestEven) {
  __int128 sum = 0;
  for (int64_t n = 0; n < extent; ++n)
    sum += ((__int128)input[n] * coefficient(extent, k, n)) >> 32;
  return saturate32(roundShift(sum, m, nearestEven));
}

// The requantized profile the same shape takes without product=raw_high:
// each term rounded nearest-even by p = m, the sum exported by 32.
static int32_t requantized(const int32_t *input, int64_t extent, int64_t k, unsigned m) {
  __int128 sum = 0;
  for (int64_t n = 0; n < extent; ++n)
    sum += roundShift((__int128)input[n] * coefficient(extent, k, n), m, 1);
  return saturate32(roundShift(sum, 32, 1));
}

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

int main(void) {
  int32_t input[64], output[64];
  uint32_t state = 0x3D9C41E7u;
  int contractsDiffer = 0;
  for (int trial = 0; trial < 8; ++trial) {
    for (int n = 0; n < 64; ++n)
      input[n] = trial == 0 ? (n & 1 ? INT32_MAX : INT32_MIN) : (int32_t)next(&state);
    ondrix_q31_dct8_raw_high(input, output);
    for (int64_t k = 0; k < 8; ++k) {
      int32_t expected = rawHigh(input, 8, k, 3, 1);
      if (output[k] != expected) {
        fprintf(stderr, "dct8 trial %d bin %lld: got %d, expected %d\n", trial, (long long)k,
                output[k], expected);
        return 1;
      }
      contractsDiffer |= expected != requantized(input, 8, k, 3);
    }
    ondrix_q31_dct64_raw_high(input, output);
    for (int64_t k = 0; k < 64; ++k) {
      int32_t expected = rawHigh(input, 64, k, 6, 0);
      if (output[k] != expected) {
        fprintf(stderr, "dct64 trial %d bin %lld: got %d, expected %d\n", trial, (long long)k,
                output[k], expected);
        return 1;
      }
    }
  }
  if (!contractsDiffer) {
    fprintf(stderr, "the corpus never separates the raw-high and requantized contracts\n");
    return 1;
  }
  return 0;
}
