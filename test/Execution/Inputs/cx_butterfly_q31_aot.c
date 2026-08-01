/* Independent reference for the packed-Q31 butterfly contract.
 *
 * This gate exists to exercise the exactness the product carrier must
 * provide. ondsp.cx_butterfly accepts an arbitrary SSA twiddle, so the cross
 * sums are bounded only by the operand widths:
 *
 *   br*wi + bi*wr <= 2^62 + 2^62 = 2^63 = 9223372036854775808
 *   INT64_MAX                            = 9223372036854775807
 *
 * The maximum is attained uniquely at b = w = (INT32_MIN, INT32_MIN) and is
 * exactly one past INT64_MAX, so a WRAPPING i64 carrier wraps to -2^63 there
 * and the shift-31 requantization saturates to INT32_MIN instead of
 * INT32_MAX. The real cross sum br*wr - bi*wi cannot overflow: its extremes
 * are +-(2^63 - 2^31), strictly inside i64. Only the imaginary term reaches
 * past i64, only in the positive direction, and only at that one corner.
 *
 * The corner therefore refutes one implementation class, not every i64-based
 * one: because 2^63 is the single unrepresentable value and the profile
 * requantizes it through a shift-31 nearest-even boundary into a saturating
 * i32, an overflow-aware i64 sum that clamps to INT64_MAX rounds to the same
 * 2^32 and saturates to the same INT32_MAX, observably equivalent everywhere.
 * The lowering's i128 is the exact generic choice (the minimal exact width
 * is i65); a saturating-i64 lowering would be admissible only with its own
 * equivalence proof, which is a statement this gate documents rather than
 * decides. */

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int64_t cx_butterfly_q31_result(int64_t a, int64_t b, int64_t twiddle, int32_t result_index);
extern int64_t cx_butterfly_q31_vector_result(int64_t a0, int64_t a1, int64_t b0, int64_t b1,
                                              int64_t w0, int64_t w1, int32_t result_index,
                                              int32_t lane);

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
 * against it lands the shift-31 remainder on exactly 2^30. What nearest-even
 * does at a tie depends on both the sign and the parity of the floor
 * quotient, so the corpus must close all four sign x parity cells:
 * 2^29 * 1518500250 = 379625062 * 2^31 + 2^30 gives a positive-even quotient
 * on the real term and (negated on the imaginary term) a negative-odd one,
 * while 3 * 2^29 against the same twiddle gives 1138875187 * 2^31 + 2^30 — a
 * positive-odd quotient, and negated a negative-even one. A rule that is
 * sign-sensitive or parity-blind fails one of the four. The CFFT corpus never
 * reaches any of them. */
enum { kQ31Sqrt2Over2 = 1518500250, kTieOperand = 1 << 29, kTieOperandOddQuotient = 3 << 29 };

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

static const struct Case *findCase(const struct Case *cases, unsigned count, const char *name) {
  for (unsigned i = 0; i < count; ++i)
    if (strcmp(cases[i].name, name) == 0)
      return &cases[i];
  fprintf(stderr, "missing directed case %s\n", name);
  exit(1);
}

/* Two independent scalar cases riding one Vector operation. Every lane and
 * every result is compared against the scalar reference of that lane alone,
 * so a lowering that shared a carrier across lanes, swapped lanes, or let one
 * lane's saturation leak into the other fails on the same directed data the
 * scalar corpus decides with. */
static int checkVectorPair(const char *name, const struct Case *lane0, const struct Case *lane1) {
  const struct Case *lanes[2] = {lane0, lane1};
  int failed = 0;
  for (int32_t result_index = 0; result_index < 2; ++result_index) {
    for (int32_t lane = 0; lane < 2; ++lane) {
      int64_t expected0, expected1;
      butterfly_reference(pack(lanes[lane]->a), pack(lanes[lane]->b), pack(lanes[lane]->twiddle),
                          &expected0, &expected1);
      int64_t expected = result_index == 0 ? expected0 : expected1;
      int64_t actual = cx_butterfly_q31_vector_result(
          pack(lanes[0]->a), pack(lanes[1]->a), pack(lanes[0]->b), pack(lanes[1]->b),
          pack(lanes[0]->twiddle), pack(lanes[1]->twiddle), result_index, lane);
      if (actual != expected) {
        fprintf(stderr,
                "%s lane %" PRId32 " out%" PRId32 ": expected %016" PRIx64 ", got %016" PRIx64 "\n",
                name, lane, result_index, (uint64_t)expected, (uint64_t)actual);
        failed = 1;
      }
    }
  }
  return failed;
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
      /* The carrier corner: imaginary cross sum is exactly 2^63, one past
       * INT64_MAX. A wrapping i64 carrier yields INT32_MIN here instead of
       * INT32_MAX, so this single case refutes that class; see the header
       * for why it does not decide against every i64 implementation. */
      {"wrapping-i64-refutation-corner", {0, 0}, {INT32_MIN, INT32_MIN}, {INT32_MIN, INT32_MIN}},
      /* The neighbouring extreme of the real term, 2^63 - 2^31, which is the
       * largest real cross sum reachable at all and still fits i64. Pinning
       * it keeps the asymmetry between the two terms under test. */
      {"real-cross-sum-maximum", {0, 0}, {INT32_MIN, INT32_MIN}, {INT32_MIN, INT32_MAX}},
      /* All four sign x parity cells of the shift-31 nearest-even tie. The
       * first pair covers positive-even and negative-odd quotients, the
       * second covers positive-odd and negative-even; negation swaps the
       * cells between the real and imaginary terms. */
      {"product-tie-even-parities", {0, 0}, {kTieOperand, 0}, {kQ31Sqrt2Over2, -kQ31Sqrt2Over2}},
      {"product-tie-even-negated", {0, 0}, {-kTieOperand, 0}, {kQ31Sqrt2Over2, -kQ31Sqrt2Over2}},
      {"product-tie-odd-parities",
       {0, 0},
       {kTieOperandOddQuotient, 0},
       {kQ31Sqrt2Over2, -kQ31Sqrt2Over2}},
      {"product-tie-odd-negated",
       {0, 0},
       {-kTieOperandOddQuotient, 0},
       {kQ31Sqrt2Over2, -kQ31Sqrt2Over2}},
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

  /* The Vector surface, on the hardest directed pairs in both lane orders:
   * the carrier corner in one lane against a tie in the other, then the
   * saturating rails against each other. */
  const unsigned caseCount = sizeof(cases) / sizeof(cases[0]);
  const struct Case *corner = findCase(cases, caseCount, "wrapping-i64-refutation-corner");
  const struct Case *tie = findCase(cases, caseCount, "product-tie-even-parities");
  const struct Case *railAdd = findCase(cases, caseCount, "rail-add");
  const struct Case *railSubtract = findCase(cases, caseCount, "rail-subtract");
  failed |= checkVectorPair("vector corner|tie", corner, tie);
  failed |= checkVectorPair("vector tie|corner", tie, corner);
  failed |= checkVectorPair("vector rails", railAdd, railSubtract);

  /* Random breadth over the full packed domain, including arbitrary twiddles
   * no CFFT stage would ever produce. */
  uint32_t state = 0x5bd1e995u;
  struct Case previous;
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
    if (trial & 1) {
      char vectorName[40];
      snprintf(vectorName, sizeof(vectorName), "vector trials %u|%u", trial - 1, trial);
      failed |= checkVectorPair(vectorName, &previous, &test);
    }
    previous = test;
    previous.name = "previous";
  }
  return failed;
}
