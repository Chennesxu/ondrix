#include <stdint.h>
#include <stdio.h>

enum { kLength = 16, kTaps = 4 };

static __int128 floorShift(__int128 value, unsigned shift) {
  __int128 quotient = value >> shift;
  return quotient;
}

// One raw-high term: the upper 32 bits of the exact product, a floor at
// frac 30; the accumulator is the 40-bit one under the declared update mode.
static __int128 accumulate(__int128 accumulator, int32_t lhs, int32_t rhs, int saturate) {
  __int128 term = floorShift((__int128)lhs * rhs, 32);
  __int128 updated = accumulator + term;
  const __int128 low = -((__int128)1 << 39), high = ((__int128)1 << 39) - 1;
  if (saturate)
    return updated < low ? low : (updated > high ? high : updated);
  __int128 wrapped = updated & (((__int128)1 << 40) - 1);
  return wrapped >= ((__int128)1 << 39) ? wrapped - ((__int128)1 << 40) : wrapped;
}

// The readout: identity export, one exact doubling, then the declared
// narrowing to i32.
static int32_t readout(__int128 accumulator, int saturate) {
  __int128 doubled = accumulator * 2;
  if (saturate)
    return (int32_t)(doubled > INT32_MAX ? INT32_MAX : (doubled < INT32_MIN ? INT32_MIN : doubled));
  return (int32_t)(uint32_t)((uint64_t)doubled & 0xFFFFFFFFu);
}

// The full-product profile the same kernel takes without product=raw_high:
// exact i64 sum at frac 62, one nearest-even export to Q31.
static int32_t fullProduct(const int32_t *lhs, const int32_t *rhs, int count) {
  __int128 sum = 0;
  for (int i = 0; i < count; ++i)
    sum += (__int128)lhs[i] * rhs[i];
  __int128 quotient = sum >> 31, remainder = sum - (quotient << 31), half = (__int128)1 << 30;
  quotient += remainder > half || (remainder == half && (quotient & 1));
  return (int32_t)(quotient > INT32_MAX ? INT32_MAX
                                        : (quotient < INT32_MIN ? INT32_MIN : quotient));
}

static uint32_t next(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

int main(void) {
  static const int32_t coefficients[kTaps] = {1073741824, -536870912, 2147483647, INT32_MIN};
  int32_t lhs[kLength], rhs[kLength];
  uint32_t state = 0x51ED270Bu;
  int floorsSeen = 0, saturationSeen = 0, wrapSeen = 0;
  for (int block = 0; block < 64; ++block) {
    for (int i = 0; i < kLength; ++i) {
      lhs[i] = (int32_t)next(&state);
      rhs[i] = (int32_t)next(&state);
    }
    if (block == 0) {
      // All-rails: every term is the largest positive raw-high value, the
      // sum passes 2^31 and the doubling must saturate.
      for (int i = 0; i < kLength; ++i)
        lhs[i] = rhs[i] = INT32_MAX;
    }
    if (block == 1) {
      // Small products: each term floors to -1 where the full product would
      // round the sum to zero.
      for (int i = 0; i < kLength; ++i) {
        lhs[i] = 3;
        rhs[i] = -5;
      }
    }
    for (int length = 1; length <= kLength; length += 5) {
      __int128 accumulator = 0;
      for (int i = 0; i < length; ++i)
        accumulator = accumulate(accumulator, lhs[i], rhs[i], 1);
      int32_t expected = readout(accumulator, 1);
      int32_t got = ondrix_q31_dot_raw_high(lhs, rhs, (uint64_t)length);
      if (got != expected) {
        fprintf(stderr, "dot block %d length %d: got %d, expected %d\n", block, length, got,
                expected);
        return 1;
      }
      floorsSeen |= expected != fullProduct(lhs, rhs, length);
      saturationSeen |= expected == INT32_MAX || expected == INT32_MIN;
    }
    __int128 accumulator = 0;
    for (int i = 0; i < kTaps; ++i)
      accumulator = accumulate(accumulator, lhs[i], coefficients[i], 0);
    int32_t expected = readout(accumulator, 0);
    int32_t got = ondrix_q31_fir_raw_high(lhs);
    if (got != expected) {
      fprintf(stderr, "fir block %d: got %d, expected %d\n", block, got, expected);
      return 1;
    }
    wrapSeen |= accumulator * 2 > INT32_MAX || accumulator * 2 < INT32_MIN;
  }
  if (!floorsSeen || !saturationSeen || !wrapSeen) {
    fprintf(stderr, "positive controls: floors %d saturation %d wrap %d\n", floorsSeen,
            saturationSeen, wrapSeen);
    return 1;
  }
  return 0;
}
