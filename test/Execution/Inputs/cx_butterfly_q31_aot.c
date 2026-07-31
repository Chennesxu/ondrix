/* Independent reference for the packed-Q31 butterfly contract.
 *
 * This gate exists to witness the i128 product carrier. ondsp.cx_butterfly
 * accepts an arbitrary SSA twiddle, so the cross sums are bounded only by the
 * operand widths:
 *
 *   br*wi + bi*wr <= 2^62 + 2^62 = 2^63 = 9223372036854775808
 *   INT64_MAX                            = 9223372036854775807
 *
 * The maximum is attained uniquely at b = w = (INT32_MIN, INT32_MIN) and is
 * exactly one past INT64_MAX, so an i64 carrier wraps to -2^63 there and the
 * shift-31 requantization saturates to INT32_MIN instead of INT32_MAX. The
 * real cross sum br*wr - bi*wi cannot overflow: its extremes are
 * +-(2^63 - 2^31), strictly inside i64. Only the imaginary term witnesses the
 * carrier, and only at that one corner. */

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

extern int64_t cx_butterfly_q31_result(int64_t a, int64_t b, int64_t twiddle, int32_t result_index);

struct Complex {
  int32_t real;
  int32_t imaginary;
};

/* Nearest-even signed right shift with saturation into i32, evaluated on the
 * exact __int128 value. Floor division plus a remainder test is total: the
 * add-half-then-shift shortcut would overflow at exactly the boundaries this
 * gate exercises. */
static int32_t requantize(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  __int128 remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }

  const __int128 half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient < INT32_MIN)
    return INT32_MIN;
  if (quotient > INT32_MAX)
    return INT32_MAX;
  return (int32_t)quotient;
}

static int32_t decodeSigned32(uint32_t bits) {
  int64_t value = bits;
  if ((bits & UINT32_C(0x80000000)) != 0)
    value -= INT64_C(4294967296);
  return (int32_t)value;
}

static int64_t toSigned64(uint64_t bits) {
  return bits < UINT64_C(0x8000000000000000)
             ? (int64_t)bits
             : (int64_t)(__int128)((__int128)bits - ((__int128)1 << 64));
}

static int64_t pack(struct Complex value) {
  return toSigned64(((uint64_t)(uint32_t)value.imaginary << 32) | (uint64_t)(uint32_t)value.real);
}

static struct Complex unpack(int64_t value) {
  return (struct Complex){decodeSigned32((uint32_t)value),
                          decodeSigned32((uint32_t)((uint64_t)value >> 32))};
}

static void butterfly_reference(int64_t packed_a, int64_t packed_b, int64_t packed_twiddle,
                                int64_t *out0, int64_t *out1) {
  struct Complex a = unpack(packed_a);
  struct Complex b = unpack(packed_b);
  struct Complex w = unpack(packed_twiddle);
  __int128 product_real = (__int128)b.real * w.real - (__int128)b.imaginary * w.imaginary;
  __int128 product_imaginary = (__int128)b.real * w.imaginary + (__int128)b.imaginary * w.real;
  int32_t twiddled_real = requantize(product_real, 31);
  int32_t twiddled_imaginary = requantize(product_imaginary, 31);

  struct Complex first = {
      requantize((__int128)a.real + twiddled_real, 1),
      requantize((__int128)a.imaginary + twiddled_imaginary, 1),
  };
  struct Complex second = {
      requantize((__int128)a.real - twiddled_real, 1),
      requantize((__int128)a.imaginary - twiddled_imaginary, 1),
  };
  *out0 = pack(first);
  *out1 = pack(second);
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

struct Case {
  const char *name;
  struct Complex a;
  struct Complex b;
  struct Complex twiddle;
};

/* 1518500250 is the frozen Q31 quantization of cos(pi/4); b.real = 2^29
 * against it lands the shift-31 remainder on exactly 2^30, with an even floor
 * quotient on the real term and an odd one on the imaginary term. That pair
 * executes both nearest-even tie parities, which the CFFT corpus never
 * reaches. */
enum { kQ31Sqrt2Over2 = 1518500250, kTieOperand = 1 << 29 };

static int check(const struct Case *test) {
  int64_t a = pack(test->a);
  int64_t b = pack(test->b);
  int64_t twiddle = pack(test->twiddle);
  int64_t expected0, expected1;
  butterfly_reference(a, b, twiddle, &expected0, &expected1);
  int64_t actual0 = cx_butterfly_q31_result(a, b, twiddle, 0);
  int64_t actual1 = cx_butterfly_q31_result(a, b, twiddle, 1);
  if (actual0 != expected0 || actual1 != expected1) {
    fprintf(stderr,
            "%s: expected (%016" PRIx64 ", %016" PRIx64 "), got (%016" PRIx64 ", %016" PRIx64 ")\n",
            test->name, (uint64_t)expected0, (uint64_t)expected1, (uint64_t)actual0,
            (uint64_t)actual1);
    return 1;
  }
  return 0;
}

int main(void) {
  static const struct Case cases[] = {
      {"zero", {0, 0}, {0, 0}, {0, 0}},
      {"identity", {800000000, -500000000}, {700000000, 900000000}, {INT32_MAX, 0}},
      {"nontrivial",
       {-1200000000, 2100000000},
       {1800000000, -1230000000},
       {kQ31Sqrt2Over2, -kQ31Sqrt2Over2}},
      {"mixed-sign", {INT32_MIN, INT32_MAX}, {-1234567890, 1234567890}, {-2000000000, 1100000000}},
      /* The i128 witness: imaginary cross sum is exactly 2^63, one past
       * INT64_MAX. A wrapping i64 carrier yields INT32_MIN here instead of
       * INT32_MAX, so this single case decides the carrier width. */
      {"i64-carrier-overflow-witness", {0, 0}, {INT32_MIN, INT32_MIN}, {INT32_MIN, INT32_MIN}},
      /* The neighbouring extreme of the real term, 2^63 - 2^31, which is the
       * largest real cross sum reachable at all and still fits i64. Pinning
       * it keeps the asymmetry between the two terms under test. */
      {"real-cross-sum-maximum", {0, 0}, {INT32_MIN, INT32_MIN}, {INT32_MIN, INT32_MAX}},
      /* Both shift-31 nearest-even tie parities in one operand pair. */
      {"product-tie-both-parities", {0, 0}, {kTieOperand, 0}, {kQ31Sqrt2Over2, -kQ31Sqrt2Over2}},
      {"product-tie-negated", {0, 0}, {-kTieOperand, 0}, {kQ31Sqrt2Over2, -kQ31Sqrt2Over2}},
      /* Output-scale ties: odd sums shifted right by one. */
      {"output-tie-odd", {1, 3}, {0, 0}, {INT32_MAX, 0}},
      {"output-tie-negative", {-1, -3}, {0, 0}, {INT32_MAX, 0}},
      /* Saturating rails through both boundaries. */
      {"rail-add", {INT32_MAX, INT32_MAX}, {INT32_MAX, INT32_MAX}, {INT32_MAX, 0}},
      {"rail-subtract", {INT32_MIN, INT32_MIN}, {INT32_MAX, INT32_MAX}, {INT32_MAX, 0}},
  };

  int failed = 0;
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    failed |= check(&cases[i]);

  /* Random breadth over the full packed domain, including arbitrary twiddles
   * no CFFT stage would ever produce. */
  uint32_t state = 0x5bd1e995u;
  for (unsigned trial = 0; trial < 64; ++trial) {
    struct Complex operand[3];
    for (unsigned i = 0; i < 3; ++i) {
      state = nextState(state);
      operand[i].real = decodeSigned32(state);
      state = nextState(state);
      operand[i].imaginary = decodeSigned32(state);
    }
    char name[32];
    snprintf(name, sizeof(name), "trial %u", trial);
    struct Case test = {name, operand[0], operand[1], operand[2]};
    failed |= check(&test);
  }
  return failed;
}
