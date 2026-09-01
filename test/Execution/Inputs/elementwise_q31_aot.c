#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

typedef void (*UnaryKernel)(MemRefI32 *, MemRefI32 *);
typedef void (*BinaryKernel)(MemRefI32 *, MemRefI32 *, MemRefI32 *);

extern void _mlir_ciface_add_saturate(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_add_wrap(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_sub_saturate(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_mult_nearest_even(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_mult_ties_positive(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_mult_toward_negative(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_mult_toward_zero(MemRefI32 *, MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_abs_saturate(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_abs_wrap(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_negate_saturate(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_negate_wrap(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_offset_saturate(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_shift_right_nearest_even(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_shift_right_ties_positive(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_shift_right_toward_negative(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_shift_right_toward_zero(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_shift_left_saturate(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_shift_left_wrap(MemRefI32 *, MemRefI32 *);

enum { kBlock = 4096, kBlocks = 16 };
enum { kOffsetBias = -1234567890 };
enum Rounding { NEAREST_EVEN, TIES_POSITIVE, TOWARD_NEGATIVE, TOWARD_ZERO };

/* Independent reference: exact integer arithmetic in int64, then the one
 * declared boundary, written out per rule rather than shared with the
 * compiler's requantizer. */
static int64_t roundShift(int64_t value, int shift, enum Rounding rounding) {
  if (shift == 0)
    return value;
  int64_t quotient = value >> shift;
  int64_t remainder = value - (quotient << shift);
  int64_t half = (int64_t)1 << (shift - 1);
  switch (rounding) {
  case NEAREST_EVEN:
    if (remainder > half || (remainder == half && (quotient & 1)))
      ++quotient;
    break;
  case TIES_POSITIVE:
    if (remainder >= half)
      ++quotient;
    break;
  case TOWARD_NEGATIVE:
    break;
  case TOWARD_ZERO:
    if (quotient < 0 && remainder != 0)
      ++quotient;
    break;
  }
  return quotient;
}

static int32_t narrow(int64_t value, int saturating) {
  if (saturating)
    return (int32_t)(value > INT32_MAX ? INT32_MAX : (value < INT32_MIN ? INT32_MIN : value));
  return (int32_t)(uint32_t)(uint64_t)value;
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int failures;

static void reportUnary(const char *label, int32_t input, int32_t got, int32_t expected) {
  if (got == expected)
    return;
  if (failures++ < 8)
    fprintf(stderr, "%s(%d): got %d, expected %d\n", label, input, got, expected);
}

static void reportBinary(const char *label, int32_t lhs, int32_t rhs, int32_t got,
                         int32_t expected) {
  if (got == expected)
    return;
  if (failures++ < 8)
    fprintf(stderr, "%s(%d, %d): got %d, expected %d\n", label, lhs, rhs, got, expected);
}

/* The 2^32 unary domain is not sweepable, so the first block is every input
 * whose answer is a boundary decision and the rest is a deterministic
 * pseudorandom spread. INT32_MIN leads because it is the input that an i32
 * body gets WRONG rather than merely imprecise, in abs, negate and both
 * shifts. */
static const int32_t kUnaryProbe[] = {INT32_MIN,   INT32_MIN + 1,
                                      INT32_MAX,   INT32_MAX - 1,
                                      0,           1,
                                      -1,          3,
                                      -3,          2,
                                      -2,          1073741824,
                                      -1073741824, 1073741823,
                                      -1073741825, 65536,
                                      -65536,      -1234567890,
                                      1234567890,  2,
                                      -4,          5,
                                      -5,          7};

static void checkUnary(UnaryKernel kernel, const char *label, const int32_t *input, int64_t count,
                       int32_t (*reference)(int32_t)) {
  MemRefI32 inputRef = {(int32_t *)input, (int32_t *)input, 0, {count}, {1}};
  MemRefI32 output;
  kernel(&output, &inputRef);
  for (int64_t i = 0; i < count; ++i)
    reportUnary(label, input[i], output.aligned[output.offset + i], reference(input[i]));
  free(output.allocated);
}

static void sweepUnary(UnaryKernel kernel, const char *label, int32_t (*reference)(int32_t)) {
  static const int64_t probeCount = (int64_t)(sizeof kUnaryProbe / sizeof kUnaryProbe[0]);
  int32_t input[kBlock];
  uint32_t state = 0x7F4A7C15u;
  for (int64_t block = 0; block < kBlocks; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      if (block == 0) {
        input[i] = kUnaryProbe[i % probeCount];
      } else {
        state = nextState(state);
        input[i] = (int32_t)state;
      }
    }
    checkUnary(kernel, label, input, kBlock, reference);
  }
}

static int32_t referenceAbsSaturate(int32_t x) {
  int64_t v = x < 0 ? -(int64_t)x : (int64_t)x;
  return narrow(v, 1);
}
static int32_t referenceAbsWrap(int32_t x) {
  int64_t v = x < 0 ? -(int64_t)x : (int64_t)x;
  return narrow(v, 0);
}
static int32_t referenceNegateSaturate(int32_t x) { return narrow(-(int64_t)x, 1); }
static int32_t referenceNegateWrap(int32_t x) { return narrow(-(int64_t)x, 0); }
static int32_t referenceOffsetSaturate(int32_t x) {
  return narrow((int64_t)x + (int64_t)kOffsetBias, 1);
}
static int32_t referenceShiftLeftSaturate(int32_t x) { return narrow((int64_t)x << 4, 1); }
static int32_t referenceShiftLeftWrap(int32_t x) { return narrow((int64_t)x << 4, 0); }
static int32_t referenceShiftRightEven(int32_t x) {
  return narrow(roundShift(x, 1, NEAREST_EVEN), 1);
}
static int32_t referenceShiftRightPositive(int32_t x) {
  return narrow(roundShift(x, 1, TIES_POSITIVE), 1);
}
static int32_t referenceShiftRightNegative(int32_t x) {
  return narrow(roundShift(x, 1, TOWARD_NEGATIVE), 1);
}
static int32_t referenceShiftRightZero(int32_t x) {
  return narrow(roundShift(x, 1, TOWARD_ZERO), 1);
}

static void checkBinary(BinaryKernel kernel, const char *label, const int32_t *lhs,
                        const int32_t *rhs, int64_t count, int32_t (*reference)(int32_t, int32_t)) {
  MemRefI32 lhsRef = {(int32_t *)lhs, (int32_t *)lhs, 0, {count}, {1}};
  MemRefI32 rhsRef = {(int32_t *)rhs, (int32_t *)rhs, 0, {count}, {1}};
  MemRefI32 output;
  kernel(&output, &lhsRef, &rhsRef);
  for (int64_t i = 0; i < count; ++i)
    reportBinary(label, lhs[i], rhs[i], output.aligned[output.offset + i],
                 reference(lhs[i], rhs[i]));
  free(output.allocated);
}

static int32_t referenceAddSaturate(int32_t a, int32_t b) {
  return narrow((int64_t)a + (int64_t)b, 1);
}
static int32_t referenceAddWrap(int32_t a, int32_t b) { return narrow((int64_t)a + (int64_t)b, 0); }
static int32_t referenceSubSaturate(int32_t a, int32_t b) {
  return narrow((int64_t)a - (int64_t)b, 1);
}
static int32_t multReference(int32_t a, int32_t b, enum Rounding rounding) {
  return narrow(roundShift((int64_t)a * (int64_t)b, 31, rounding), 1);
}
static int32_t referenceMultEven(int32_t a, int32_t b) { return multReference(a, b, NEAREST_EVEN); }
static int32_t referenceMultPositive(int32_t a, int32_t b) {
  return multReference(a, b, TIES_POSITIVE);
}
static int32_t referenceMultNegative(int32_t a, int32_t b) {
  return multReference(a, b, TOWARD_NEGATIVE);
}
static int32_t referenceMultZero(int32_t a, int32_t b) { return multReference(a, b, TOWARD_ZERO); }

/* The four tie rules are only a contract if they disagree somewhere the gate
 * actually runs. At a right shift of one the inputs 1, 3, -1 and -3 separate
 * all four completely, which no single positive input does. */
static int checkShiftRuleSeparation(void) {
  static const int32_t probe[4] = {1, 3, -1, -3};
  int32_t input[kBlock];
  for (int64_t i = 0; i < kBlock; ++i)
    input[i] = probe[i & 3];
  MemRefI32 inputRef = {input, input, 0, {kBlock}, {1}};
  MemRefI32 results[4];
  _mlir_ciface_shift_right_nearest_even(&results[0], &inputRef);
  _mlir_ciface_shift_right_ties_positive(&results[1], &inputRef);
  _mlir_ciface_shift_right_toward_negative(&results[2], &inputRef);
  _mlir_ciface_shift_right_toward_zero(&results[3], &inputRef);
  int distinct = 1;
  for (int a = 0; a < 4 && distinct; ++a)
    for (int b = a + 1; b < 4 && distinct; ++b) {
      int differs = 0;
      for (int64_t i = 0; i < 4; ++i)
        if (results[a].aligned[results[a].offset + i] != results[b].aligned[results[b].offset + i])
          differs = 1;
      if (!differs) {
        fprintf(stderr, "shift rounding rules %d and %d agree on the separating probe\n", a, b);
        distinct = 0;
      }
    }
  for (int i = 0; i < 4; ++i)
    free(results[i].allocated);
  return !distinct;
}

/* The three answers an i32 body cannot produce. Asserted by value here, not
 * only through the reference, so the carrier claim reads as a claim. */
static int checkWideCarrier(void) {
  int32_t input[kBlock];
  int failed = 0;
  for (int64_t i = 0; i < kBlock; ++i)
    input[i] = INT32_MIN;
  MemRefI32 inputRef = {input, input, 0, {kBlock}, {1}};
  MemRefI32 product;
  _mlir_ciface_mult_nearest_even(&product, &inputRef, &inputRef);
  if (product.aligned[product.offset] != INT32_MAX) {
    fprintf(stderr, "mult(INT32_MIN, INT32_MIN): got %d, expected %d\n",
            product.aligned[product.offset], INT32_MAX);
    failed = 1;
  }
  free(product.allocated);
  MemRefI32 magnitude;
  _mlir_ciface_abs_saturate(&magnitude, &inputRef);
  if (magnitude.aligned[magnitude.offset] != INT32_MAX) {
    fprintf(stderr, "abs_saturate(INT32_MIN): got %d, expected %d\n",
            magnitude.aligned[magnitude.offset], INT32_MAX);
    failed = 1;
  }
  free(magnitude.allocated);
  for (int64_t i = 0; i < kBlock; ++i)
    input[i] = INT32_MAX;
  MemRefI32 shifted;
  _mlir_ciface_shift_left_saturate(&shifted, &inputRef);
  if (shifted.aligned[shifted.offset] != INT32_MAX) {
    fprintf(stderr, "shift_left_saturate(INT32_MAX): got %d, expected %d\n",
            shifted.aligned[shifted.offset], INT32_MAX);
    failed = 1;
  }
  free(shifted.allocated);
  return failed;
}

int main(void) {
  sweepUnary(_mlir_ciface_abs_saturate, "abs_saturate", referenceAbsSaturate);
  sweepUnary(_mlir_ciface_abs_wrap, "abs_wrap", referenceAbsWrap);
  sweepUnary(_mlir_ciface_negate_saturate, "negate_saturate", referenceNegateSaturate);
  sweepUnary(_mlir_ciface_negate_wrap, "negate_wrap", referenceNegateWrap);
  sweepUnary(_mlir_ciface_offset_saturate, "offset_saturate", referenceOffsetSaturate);
  sweepUnary(_mlir_ciface_shift_left_saturate, "shift_left_saturate", referenceShiftLeftSaturate);
  sweepUnary(_mlir_ciface_shift_left_wrap, "shift_left_wrap", referenceShiftLeftWrap);
  sweepUnary(_mlir_ciface_shift_right_nearest_even, "shift_right_even", referenceShiftRightEven);
  sweepUnary(_mlir_ciface_shift_right_ties_positive, "shift_right_positive",
             referenceShiftRightPositive);
  sweepUnary(_mlir_ciface_shift_right_toward_negative, "shift_right_negative",
             referenceShiftRightNegative);
  sweepUnary(_mlir_ciface_shift_right_toward_zero, "shift_right_zero", referenceShiftRightZero);

  int32_t lhs[kBlock];
  int32_t rhs[kBlock];
  uint32_t state = 0x3D9A61F7u;
  for (int64_t block = 0; block < kBlocks; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      state = nextState(state);
      lhs[i] = (int32_t)state;
      state = nextState(state);
      rhs[i] = (int32_t)state;
    }
    if (block == 0)
      /* Every rail pair, including INT32_MIN * INT32_MIN = +1.0, the one
       * product the destination cannot hold. */
      for (int64_t i = 0; i < kBlock; ++i) {
        lhs[i] = (i & 1) ? INT32_MIN : INT32_MAX;
        rhs[i] = (i & 2) ? INT32_MIN : INT32_MAX;
      }
    if (block == 1)
      /* Halfway products at both signs. The negative one is the separator:
       * -2^30 * 3 has floor quotient -2 (even), where nearest_even stays
       * and nearest_ties_positive steps up; the positive one has floor
       * quotient 1 (odd), where the two rules agree. */
      for (int64_t i = 0; i < kBlock; ++i) {
        lhs[i] = (i & 1) ? -1073741824 : 1073741824;
        rhs[i] = 3;
      }
    checkBinary(_mlir_ciface_add_saturate, "add_saturate", lhs, rhs, kBlock, referenceAddSaturate);
    checkBinary(_mlir_ciface_add_wrap, "add_wrap", lhs, rhs, kBlock, referenceAddWrap);
    checkBinary(_mlir_ciface_sub_saturate, "sub_saturate", lhs, rhs, kBlock, referenceSubSaturate);
    checkBinary(_mlir_ciface_mult_nearest_even, "mult_even", lhs, rhs, kBlock, referenceMultEven);
    checkBinary(_mlir_ciface_mult_ties_positive, "mult_positive", lhs, rhs, kBlock,
                referenceMultPositive);
    checkBinary(_mlir_ciface_mult_toward_negative, "mult_negative", lhs, rhs, kBlock,
                referenceMultNegative);
    checkBinary(_mlir_ciface_mult_toward_zero, "mult_zero", lhs, rhs, kBlock, referenceMultZero);
  }

  return failures != 0 || checkShiftRuleSeparation() || checkWideCarrier();
}
