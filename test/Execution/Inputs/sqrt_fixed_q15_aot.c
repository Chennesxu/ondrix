#include <stdint.h>
#include <stdio.h>

extern int16_t sqrt_nearest_q15(int64_t input);
extern int16_t sqrt_floor_q15(int64_t input);

/* Independent contract reference: 32-candidate-bit unsigned integer square
 * root over the full non-negative i64 domain. Because the input is bounded
 * by 2^63 - 1, every accepted root is at most 3037000499 and every candidate
 * square fits uint64_t. Negative inputs are outside the value domain; the
 * lowering pins them to the deterministic result 0 and the reference
 * mirrors that clamp. */
static int16_t referenceSqrt(int64_t input, int nearest) {
  if (input < 0)
    return 0;
  uint64_t s = (uint64_t)input;
  uint64_t root = 0;
  for (int bit = 31; bit >= 0; --bit) {
    uint64_t candidate = root | ((uint64_t)1 << bit);
    if (candidate * candidate <= s)
      root = candidate;
  }
  if (nearest && s - root * root > root)
    ++root;
  if (root > 32767)
    root = 32767;
  return (int16_t)root;
}

static uint64_t nextState(uint64_t state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

static int checkOne(int64_t input) {
  int failed = 0;
  int16_t nearest = sqrt_nearest_q15(input);
  int16_t nearestExpected = referenceSqrt(input, 1);
  if (nearest != nearestExpected) {
    fprintf(stderr, "nearest(%lld): got %d, expected %d\n", (long long)input, nearest,
            nearestExpected);
    failed = 1;
  }
  int16_t floor16 = sqrt_floor_q15(input);
  int16_t floorExpected = referenceSqrt(input, 0);
  if (floor16 != floorExpected) {
    fprintf(stderr, "floor(%lld): got %d, expected %d\n", (long long)input, floor16, floorExpected);
    failed = 1;
  }
  return failed;
}

/* Roots whose square neighborhoods carry the rounding and saturation
 * decisions: small values, the pre-/post-saturation region around
 * 32767/32768, the 16-bit boundary around 65535/65536/65537 (the estimate
 * clamp ceiling), and the largest root reachable from the i64 domain. */
static const uint64_t kRoots[] = {1,
                                  2,
                                  3,
                                  4,
                                  181,
                                  255,
                                  256,
                                  257,
                                  16384,
                                  32766,
                                  32767,
                                  32768,
                                  32769,
                                  46340,
                                  46341,
                                  65535,
                                  65536,
                                  65537,
                                  1000000,
                                  94906265,
                                  94906266,
                                  /* floor(sqrt(2^63 - 1)) */ 3037000499u};

int main(void) {
  int failed = 0;

  failed |= checkOne(0);

  for (unsigned i = 0; i < sizeof kRoots / sizeof kRoots[0]; ++i) {
    uint64_t r = kRoots[i];
    uint64_t square = r * r;
    failed |= checkOne((int64_t)(square - 1));
    failed |= checkOne((int64_t)square);
    failed |= checkOne((int64_t)(square + 1));
    failed |= checkOne((int64_t)(square + r));     /* last input rounding down */
    failed |= checkOne((int64_t)(square + r + 1)); /* first input rounding up */
    uint64_t nextSquareMinusOne = square + 2 * r;  /* (r+1)^2 - 1 */
    if (nextSquareMinusOne <= (uint64_t)INT64_MAX)
      failed |= checkOne((int64_t)nextSquareMinusOne);
  }

  /* Domain-wide boundaries independent of a specific root. */
  static const int64_t kBoundaries[] = {
      INT64_C(2147483647), /* 2^31 - 1: old proven producer bound */
      INT64_C(2147483648), /* 2^31 */
      INT64_C(4294967295), /* 2^32 - 1: last exact-case input */
      INT64_C(4294967296), /* 2^32: first estimate-ceiling input */
      INT64_C(4294967297),
      INT64_C(9007199254740991), /* 2^53 - 1: last exact binary64 i64 */
      INT64_C(9007199254740992), /* 2^53 */
      INT64_C(9007199254740993), /* 2^53 + 1: first inexact conversion */
      INT64_C(9007199254740994),
      INT64_MAX - 1,
      INT64_MAX,
  };
  for (unsigned i = 0; i < sizeof kBoundaries / sizeof kBoundaries[0]; ++i)
    failed |= checkOne(kBoundaries[i]);

  /* Out-of-domain runtime values: the deterministic clamp yields 0. */
  static const int64_t kNegatives[] = {
      -1, -2, -32768, INT64_C(-2147483648), INT64_C(-4611686018427387904), INT64_MIN};
  for (unsigned i = 0; i < sizeof kNegatives / sizeof kNegatives[0]; ++i) {
    if (sqrt_nearest_q15(kNegatives[i]) != 0 || sqrt_floor_q15(kNegatives[i]) != 0) {
      fprintf(stderr, "negative %lld did not clamp to 0\n", (long long)kNegatives[i]);
      failed = 1;
    }
  }

  /* Deterministic random sweep over the full non-negative domain. */
  uint64_t state = UINT64_C(0x51D5A15E) << 17 | 0x0A15;
  for (int trial = 0; trial < 256; ++trial) {
    state = nextState(state);
    failed |= checkOne((int64_t)(state >> 1));
  }
  return failed;
}
