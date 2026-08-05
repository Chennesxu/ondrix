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

typedef void (*UnaryKernel)(MemRefI16 *, MemRefI16 *);
typedef void (*BinaryKernel)(MemRefI16 *, MemRefI16 *, MemRefI16 *);

extern void _mlir_ciface_add_saturate(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_add_wrap(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_sub_saturate(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_mult_nearest_even(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_mult_ties_positive(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_mult_toward_negative(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_mult_toward_zero(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_abs_saturate(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_abs_wrap(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_negate_saturate(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_negate_wrap(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_offset_saturate(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_right_nearest_even(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_right_ties_positive(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_right_toward_negative(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_right_toward_zero(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_left_saturate(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_left_wrap(MemRefI16 *, MemRefI16 *);

enum { kBlock = 4096, kBlocks = 16, kOffsetBias = -12345 };
enum Rounding { NEAREST_EVEN, TIES_POSITIVE, TOWARD_NEGATIVE, TOWARD_ZERO };

/* Independent reference: exact integer arithmetic in int32, then the one
 * declared boundary, written out per rule rather than shared with the
 * compiler's requantizer. */
static int32_t roundShift(int32_t value, int shift, enum Rounding rounding) {
  if (shift == 0)
    return value;
  int32_t quotient = value >> shift;
  int32_t remainder = value - (quotient << shift);
  int32_t half = (int32_t)1 << (shift - 1);
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

static int16_t narrow(int32_t value, int saturating) {
  if (saturating)
    return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
  return (int16_t)(uint16_t)(uint32_t)value;
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int failures;

static void reportUnary(const char *label, int16_t input, int16_t got, int16_t expected) {
  if (got == expected)
    return;
  if (failures++ < 8)
    fprintf(stderr, "%s(%d): got %d, expected %d\n", label, input, got, expected);
}

static void reportBinary(const char *label, int16_t lhs, int16_t rhs, int16_t got,
                         int16_t expected) {
  if (got == expected)
    return;
  if (failures++ < 8)
    fprintf(stderr, "%s(%d, %d): got %d, expected %d\n", label, lhs, rhs, got, expected);
}

/* The unary domain is 2^16 values, so it is swept whole rather than sampled:
 * the -32768 rail, every rounding tie, and every wrap are all reached by
 * construction rather than by hoping a corpus lands on them. */
static void sweepUnary(UnaryKernel kernel, const char *label, int16_t (*reference)(int16_t)) {
  int16_t input[kBlock];
  for (int64_t block = 0; block < kBlocks; ++block) {
    for (int64_t i = 0; i < kBlock; ++i)
      input[i] = (int16_t)(int32_t)(block * kBlock + i - 32768);
    MemRefI16 inputRef = {input, input, 0, {kBlock}, {1}};
    MemRefI16 output;
    kernel(&output, &inputRef);
    for (int64_t i = 0; i < kBlock; ++i)
      reportUnary(label, input[i], output.aligned[output.offset + i], reference(input[i]));
    free(output.allocated);
  }
}

static int16_t referenceAbsSaturate(int16_t x) {
  int32_t v = x < 0 ? -(int32_t)x : (int32_t)x;
  return narrow(v, 1);
}
static int16_t referenceAbsWrap(int16_t x) {
  int32_t v = x < 0 ? -(int32_t)x : (int32_t)x;
  return narrow(v, 0);
}
static int16_t referenceNegateSaturate(int16_t x) { return narrow(-(int32_t)x, 1); }
static int16_t referenceNegateWrap(int16_t x) { return narrow(-(int32_t)x, 0); }
static int16_t referenceOffsetSaturate(int16_t x) { return narrow((int32_t)x + kOffsetBias, 1); }
static int16_t referenceShiftLeftSaturate(int16_t x) { return narrow((int32_t)x << 4, 1); }
static int16_t referenceShiftLeftWrap(int16_t x) { return narrow((int32_t)x << 4, 0); }
static int16_t referenceShiftRightEven(int16_t x) {
  return narrow(roundShift(x, 1, NEAREST_EVEN), 1);
}
static int16_t referenceShiftRightPositive(int16_t x) {
  return narrow(roundShift(x, 1, TIES_POSITIVE), 1);
}
static int16_t referenceShiftRightNegative(int16_t x) {
  return narrow(roundShift(x, 1, TOWARD_NEGATIVE), 1);
}
static int16_t referenceShiftRightZero(int16_t x) {
  return narrow(roundShift(x, 1, TOWARD_ZERO), 1);
}

static void checkBinary(BinaryKernel kernel, const char *label, const int16_t *lhs,
                        const int16_t *rhs, int64_t count, int16_t (*reference)(int16_t, int16_t)) {
  MemRefI16 lhsRef = {(int16_t *)lhs, (int16_t *)lhs, 0, {count}, {1}};
  MemRefI16 rhsRef = {(int16_t *)rhs, (int16_t *)rhs, 0, {count}, {1}};
  MemRefI16 output;
  kernel(&output, &lhsRef, &rhsRef);
  for (int64_t i = 0; i < count; ++i)
    reportBinary(label, lhs[i], rhs[i], output.aligned[output.offset + i],
                 reference(lhs[i], rhs[i]));
  free(output.allocated);
}

static int16_t referenceAddSaturate(int16_t a, int16_t b) {
  return narrow((int32_t)a + (int32_t)b, 1);
}
static int16_t referenceAddWrap(int16_t a, int16_t b) { return narrow((int32_t)a + (int32_t)b, 0); }
static int16_t referenceSubSaturate(int16_t a, int16_t b) {
  return narrow((int32_t)a - (int32_t)b, 1);
}
static int16_t multReference(int16_t a, int16_t b, enum Rounding rounding) {
  return narrow(roundShift((int32_t)a * (int32_t)b, 15, rounding), 1);
}
static int16_t referenceMultEven(int16_t a, int16_t b) { return multReference(a, b, NEAREST_EVEN); }
static int16_t referenceMultPositive(int16_t a, int16_t b) {
  return multReference(a, b, TIES_POSITIVE);
}
static int16_t referenceMultNegative(int16_t a, int16_t b) {
  return multReference(a, b, TOWARD_NEGATIVE);
}
static int16_t referenceMultZero(int16_t a, int16_t b) { return multReference(a, b, TOWARD_ZERO); }

/* The four tie rules are only a contract if they disagree somewhere the gate
 * actually runs. At a right shift of one the inputs 1, 3, -1 and -3 separate
 * all four completely, which no single positive input does. */
static int checkShiftRuleSeparation(void) {
  static const int16_t probe[4] = {1, 3, -1, -3};
  int16_t input[kBlock];
  for (int64_t i = 0; i < kBlock; ++i)
    input[i] = probe[i & 3];
  MemRefI16 inputRef = {input, input, 0, {kBlock}, {1}};
  MemRefI16 results[4];
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

  /* The binary domain is 2^32, so it is covered by the rails, the reachable
   * halfway products, and a deterministic random sweep instead. */
  int16_t lhs[kBlock];
  int16_t rhs[kBlock];
  uint32_t state = 0x3D9A61F7u;
  for (int64_t block = 0; block < kBlocks; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      state = nextState(state);
      lhs[i] = (int16_t)(state & 0xFFFFu);
      state = nextState(state);
      rhs[i] = (int16_t)(state & 0xFFFFu);
    }
    if (block == 0)
      /* Every rail pair, including -32768 * -32768 = +1.0, the one product
       * the destination cannot hold. */
      for (int64_t i = 0; i < kBlock; ++i) {
        lhs[i] = (i & 1) ? INT16_MIN : INT16_MAX;
        rhs[i] = (i & 2) ? INT16_MIN : INT16_MAX;
      }
    if (block == 1)
      /* Halfway products at both signs. The negative one is the separator:
       * -16384 * 3 has floor quotient -2 (even), where nearest_even stays
       * and nearest_ties_positive steps up; the positive one has floor
       * quotient 1 (odd), where the two rules agree. */
      for (int64_t i = 0; i < kBlock; ++i) {
        lhs[i] = (int16_t)((i & 1) ? -16384 : 16384);
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

  return failures != 0 || checkShiftRuleSeparation();
}
