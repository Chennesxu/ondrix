/* Independent reference for the ondsp.round_div contract.
 *
 * Written from the contract text: exact pre-scale in a wide carrier,
 * Euclidean floor quotient with a non-negative remainder, one increment
 * decided per rounding mode over (q, r) with the half test stated against
 * divisor - r (never 2r), then the declared narrowing. Every intermediate
 * is __int128 so the i112-carrier profile is exercised exactly. */

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

extern int16_t rd16_d3_floor(int16_t);
extern int16_t rd16_d3_zero(int16_t);
extern int16_t rd16_d3_tp(int16_t);
extern int16_t rd16_d3_ne(int16_t);
extern int16_t rd16_d6_floor(int16_t);
extern int16_t rd16_d6_zero(int16_t);
extern int16_t rd16_d6_tp(int16_t);
extern int16_t rd16_d6_ne(int16_t);
extern int16_t rd16_d1_ne(int16_t);
extern int16_t rd16_dmax_ne(int16_t);
extern int16_t rd16_d4_ne(int16_t);
extern int16_t rs16_s2_ne(int16_t);
extern int16_t rd32_d5_ne_sat16(int32_t);
extern int16_t rd32_d5_tp_wrap16(int32_t);
extern int32_t rd32_d6_ne(int32_t);
extern int32_t rd64_prescaled(int64_t);

enum Mode { kFloor, kZero, kTiesPositive, kNearestEven };
enum Overflow { kSaturate, kWrap };

static int64_t reference(int64_t x, int64_t d, unsigned preShift, enum Mode mode,
                         enum Overflow overflow, unsigned resultWidth) {
  __int128 scaled = (__int128)x << preShift;
  __int128 quotient = scaled / d;
  __int128 remainder = scaled % d;
  if (remainder < 0) {
    --quotient;
    remainder += d;
  }
  __int128 rounded;
  switch (mode) {
  case kFloor:
    rounded = quotient;
    break;
  case kZero:
    rounded = quotient + (scaled < 0 && remainder != 0 ? 1 : 0);
    break;
  case kTiesPositive:
    rounded = quotient + (remainder >= d - remainder ? 1 : 0);
    break;
  case kNearestEven:
  default:
    rounded =
        quotient +
        ((remainder > d - remainder || (remainder == d - remainder && (quotient & 1))) ? 1 : 0);
    break;
  }
  __int128 low = -((__int128)1 << (resultWidth - 1));
  __int128 high = ((__int128)1 << (resultWidth - 1)) - 1;
  if (overflow == kSaturate) {
    if (rounded < low)
      return (int64_t)low;
    if (rounded > high)
      return (int64_t)high;
    return (int64_t)rounded;
  }
  /* Wrap: two's-complement truncation to resultWidth. */
  uint64_t mask = resultWidth == 64 ? ~UINT64_C(0) : ((UINT64_C(1) << resultWidth) - 1);
  uint64_t bits = (uint64_t)rounded & mask;
  if (resultWidth < 64 && (bits & (UINT64_C(1) << (resultWidth - 1))))
    bits |= ~mask;
  return (int64_t)bits;
}

static int failed;

static void expectEqual(const char *label, int64_t input, int64_t actual, int64_t expected) {
  if (actual != expected) {
    fprintf(stderr, "%s(%" PRId64 "): got %" PRId64 ", expected %" PRId64 "\n", label, input,
            actual, expected);
    failed = 1;
  }
}

struct SweptConfig {
  const char *label;
  int16_t (*kernel)(int16_t);
  int64_t divisor;
  enum Mode mode;
};

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

int main(void) {
  /* Exhaustive i16 sweeps. The odd divisor never reaches a tie; inside the
   * even divisor's sweep every sign x parity tie cell occurs (remainder 3
   * of 6 with even and odd quotients of both signs). */
  static const struct SweptConfig configs[] = {
      {"rd16_d3_floor", rd16_d3_floor, 3, kFloor},
      {"rd16_d3_zero", rd16_d3_zero, 3, kZero},
      {"rd16_d3_tp", rd16_d3_tp, 3, kTiesPositive},
      {"rd16_d3_ne", rd16_d3_ne, 3, kNearestEven},
      {"rd16_d6_floor", rd16_d6_floor, 6, kFloor},
      {"rd16_d6_zero", rd16_d6_zero, 6, kZero},
      {"rd16_d6_tp", rd16_d6_tp, 6, kTiesPositive},
      {"rd16_d6_ne", rd16_d6_ne, 6, kNearestEven},
      {"rd16_d1_ne", rd16_d1_ne, 1, kNearestEven},
      {"rd16_dmax_ne", rd16_dmax_ne, 32767, kNearestEven},
      {"rd16_d4_ne", rd16_d4_ne, 4, kNearestEven},
  };
  for (unsigned config = 0; config < sizeof(configs) / sizeof(configs[0]); ++config) {
    for (int32_t raw = INT16_MIN; raw <= INT16_MAX; ++raw) {
      int16_t x = (int16_t)raw;
      expectEqual(configs[config].label, x, configs[config].kernel(x),
                  reference(x, configs[config].divisor, 0, configs[config].mode, kSaturate, 16));
    }
  }

  /* The power-of-two divisor and the shift spelling of the same declared
   * boundary must agree bit-exactly, input by input. */
  for (int32_t raw = INT16_MIN; raw <= INT16_MAX; ++raw) {
    int16_t x = (int16_t)raw;
    expectEqual("rd16_d4_ne vs rs16_s2_ne", x, rd16_d4_ne(x), rs16_s2_ne(x));
  }

  /* Directed i32 corpus: both saturating rails, the wrap narrowing, the
   * even-divisor tie quadrant (15, 21, -15, -21 land remainder 3 of 6 with
   * even/odd quotients of both signs), and the extreme numerators. */
  static const int32_t directed32[] = {0,
                                       1,
                                       -1,
                                       2,
                                       -2,
                                       3,
                                       -3,
                                       15,
                                       21,
                                       -15,
                                       -21,
                                       163840,
                                       -163840,
                                       INT32_MAX,
                                       INT32_MIN,
                                       INT32_MAX - 1,
                                       INT32_MIN + 1};
  for (unsigned i = 0; i < sizeof(directed32) / sizeof(directed32[0]); ++i) {
    int32_t x = directed32[i];
    expectEqual("rd32_d5_ne_sat16", x, rd32_d5_ne_sat16(x),
                reference(x, 5, 0, kNearestEven, kSaturate, 16));
    expectEqual("rd32_d5_tp_wrap16", x, rd32_d5_tp_wrap16(x),
                reference(x, 5, 0, kTiesPositive, kWrap, 16));
    expectEqual("rd32_d6_ne", x, rd32_d6_ne(x), reference(x, 6, 0, kNearestEven, kSaturate, 32));
  }

  /* The pre-scaled i112-carrier profile: the shift happens after widening,
   * so INT64_MIN << 48 is exact and the division runs past 64 bits. */
  static const int64_t directed64[] = {
      0, 1, -1, 999999937, -999999937, INT64_MAX, INT64_MIN, INT64_MAX - 1, INT64_MIN + 1};
  for (unsigned i = 0; i < sizeof(directed64) / sizeof(directed64[0]); ++i) {
    int64_t x = directed64[i];
    expectEqual("rd64_prescaled", x, rd64_prescaled(x),
                reference(x, 999999937, 48, kNearestEven, kSaturate, 32));
  }
  uint32_t state = 0x2545f491u;
  for (unsigned trial = 0; trial < 256; ++trial) {
    state = nextState(state);
    uint32_t high = state;
    state = nextState(state);
    int64_t x = (int64_t)(((uint64_t)high << 32) | state);
    expectEqual("rd64_prescaled trial", x, rd64_prescaled(x),
                reference(x, 999999937, 48, kNearestEven, kSaturate, 32));
  }

  return failed;
}
