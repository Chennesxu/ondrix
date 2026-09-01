/* Independent matmul reference over both fixed profiles.
 *
 * It derives the product shift from the inner extent rather than taking it as
 * a constant, and it reaches each boundary through an explicit floor-division
 * form rather than the shift the lowering emits. The Q15 arms differ only in
 * the declared export rounding; the harness REQUIRES them to disagree on at
 * least one element, which is what a pinned rounding would fail.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRefI32x2;

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRefI16x2;

extern void _mlir_ciface_matmul_k64_q31(MemRefI32x2 *, MemRefI32x2 *, MemRefI32x2 *);
extern void _mlir_ciface_matmul_k5_q31_floor(MemRefI32x2 *, MemRefI32x2 *, MemRefI32x2 *);
extern void _mlir_ciface_matmul_k8_q15_even(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);
extern void _mlir_ciface_matmul_k8_q15_floor(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);

enum { kEven = 0, kFloor = 1, kTiesPositive = 2 };

/* The smallest shift keeping a sum of `inner` products of signed Q1.(W-1)
 * operands exact in i64: 2*(W-1) + floor(log2 inner) <= 62. It is floor, not
 * ceil -- `inner` terms bounded by 2^(2*(W-1)-p) sum below
 * 2^(2*(W-1)-p+floor(log2 inner)+1), so ceil spends a bit of precision at
 * every non-power-of-two `inner`. */
static unsigned productShift(unsigned storageWidth, int64_t inner) {
  unsigned bits = 0;
  while (((int64_t)1 << (bits + 1)) <= inner)
    ++bits;
  unsigned exact = 2u * (storageWidth - 1u) + bits;
  return exact <= 62u ? 0u : exact - 62u;
}

static int64_t roundShift(int64_t value, unsigned shift, int mode) {
  if (shift == 0)
    return value;
  int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  if (mode == kFloor)
    return quotient;
  int64_t half = divisor >> 1;
  if (remainder > half)
    return quotient + 1;
  if (remainder < half)
    return quotient;
  if (mode == kTiesPositive)
    return quotient + 1;
  return (quotient & 1) ? quotient + 1 : quotient;
}

static int64_t saturate(int64_t value, unsigned storageWidth) {
  int64_t top = ((int64_t)1 << (storageWidth - 1)) - 1;
  int64_t bottom = -((int64_t)1 << (storageWidth - 1));
  return value > top ? top : (value < bottom ? bottom : value);
}

static int64_t reference(const int64_t *a, const int64_t *b, int64_t inner, int64_t columns,
                         int64_t row, int64_t column, unsigned storageWidth, int productMode,
                         int exportMode) {
  unsigned shift = productShift(storageWidth, inner);
  int64_t sum = 0;
  for (int64_t k = 0; k < inner; ++k)
    sum += roundShift(a[row * inner + k] * b[k * columns + column], shift, productMode);
  return saturate(roundShift(sum, storageWidth - 1 - shift, exportMode), storageWidth);
}

static int failures = 0;
static uint32_t state = 0x2468acedu;

static int32_t nextRandom(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (int32_t)state;
}

static void checkQ31(const char *name, void (*kernel)(MemRefI32x2 *, MemRefI32x2 *, MemRefI32x2 *),
                     int32_t *a, int32_t *b, int64_t inner, int productMode, int exportMode) {
  enum { kRows = 3, kColumns = 3 };
  MemRefI32x2 lhs = {a, a, 0, {kRows, inner}, {inner, 1}};
  MemRefI32x2 rhs = {b, b, 0, {inner, kColumns}, {kColumns, 1}};
  MemRefI32x2 out;
  kernel(&out, &lhs, &rhs);
  int64_t wa[3 * 64], wb[64 * 3];
  for (int64_t i = 0; i < kRows * inner; ++i)
    wa[i] = a[i];
  for (int64_t i = 0; i < inner * kColumns; ++i)
    wb[i] = b[i];
  for (int64_t i = 0; i < kRows; ++i)
    for (int64_t j = 0; j < kColumns; ++j) {
      int32_t observed = out.aligned[out.offset + i * out.strides[0] + j * out.strides[1]];
      int64_t expected = reference(wa, wb, inner, kColumns, i, j, 32, productMode, exportMode);
      if (observed != expected) {
        printf("%s[%lld][%lld]: observed %d expected %lld\n", name, (long long)i, (long long)j,
               observed, (long long)expected);
        ++failures;
      }
    }
  free(out.allocated);
}

static void checkQ15(const char *name, void (*kernel)(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *),
                     int16_t *a, int16_t *b, int exportMode, int16_t *observedOut) {
  enum { kRows = 3, kInner = 8, kColumns = 3 };
  MemRefI16x2 lhs = {a, a, 0, {kRows, kInner}, {kInner, 1}};
  MemRefI16x2 rhs = {b, b, 0, {kInner, kColumns}, {kColumns, 1}};
  MemRefI16x2 out;
  kernel(&out, &lhs, &rhs);
  int64_t wa[kRows * kInner], wb[kInner * kColumns];
  for (int64_t i = 0; i < kRows * kInner; ++i)
    wa[i] = a[i];
  for (int64_t i = 0; i < kInner * kColumns; ++i)
    wb[i] = b[i];
  for (int64_t i = 0; i < kRows; ++i)
    for (int64_t j = 0; j < kColumns; ++j) {
      int16_t observed = out.aligned[out.offset + i * out.strides[0] + j * out.strides[1]];
      int64_t expected = reference(wa, wb, kInner, kColumns, i, j, 16, kEven, exportMode);
      observedOut[i * kColumns + j] = observed;
      if (observed != expected) {
        printf("%s[%lld][%lld]: observed %d expected %lld\n", name, (long long)i, (long long)j,
               observed, (long long)expected);
        ++failures;
      }
    }
  free(out.allocated);
}

int main(void) {
  static int32_t a31[3 * 64], b31[64 * 3];
  static int16_t a15[3 * 8], b15[8 * 3];
  int roundingArmsDiffered = 0;

  const int32_t rails[] = {0, INT32_MAX, INT32_MIN, 1, -1, 1 << 30};
  for (unsigned rail = 0; rail < sizeof(rails) / sizeof(rails[0]); ++rail) {
    for (int64_t i = 0; i < 3 * 64; ++i)
      a31[i] = rails[rail];
    for (int64_t i = 0; i < 64 * 3; ++i)
      b31[i] = rails[rail];
    checkQ31("k64_rail", _mlir_ciface_matmul_k64_q31, a31, b31, 64, kEven, kEven);
    checkQ31("k5_rail", _mlir_ciface_matmul_k5_q31_floor, a31, b31, 5, kFloor, kFloor);
  }

  /* A named double-rounding witness for the non-power-of-two shift, because a
   * random corpus does not reach it: at K=5 the contract's p=2 exports 1 here
   * and a ceil-derived p=3 exports 0, so an oracle that derives the shift the
   * other way passes every random trial and still proves nothing. */
  for (int64_t i = 0; i < 3 * 5; ++i)
    a31[i] = 0;
  for (int64_t i = 0; i < 5 * 3; ++i)
    b31[i] = 0;
  a31[0] = 4;
  a31[1] = 2147483644;
  b31[0] = 1;
  b31[3] = 1;
  checkQ31("k5_double_rounding", _mlir_ciface_matmul_k5_q31_floor, a31, b31, 5, kFloor, kFloor);

  for (int trial = 0; trial < 200; ++trial) {
    for (int64_t i = 0; i < 3 * 64; ++i)
      a31[i] = nextRandom();
    for (int64_t i = 0; i < 64 * 3; ++i)
      b31[i] = nextRandom();
    checkQ31("k64", _mlir_ciface_matmul_k64_q31, a31, b31, 64, kEven, kEven);
    checkQ31("k5", _mlir_ciface_matmul_k5_q31_floor, a31, b31, 5, kFloor, kFloor);

    for (int64_t i = 0; i < 3 * 8; ++i)
      a15[i] = (int16_t)nextRandom();
    for (int64_t i = 0; i < 8 * 3; ++i)
      b15[i] = (int16_t)nextRandom();
    int16_t even[9], floorArm[9];
    checkQ15("q15_even", _mlir_ciface_matmul_k8_q15_even, a15, b15, kEven, even);
    checkQ15("q15_floor", _mlir_ciface_matmul_k8_q15_floor, a15, b15, kFloor, floorArm);
    for (int i = 0; i < 9; ++i)
      if (even[i] != floorArm[i])
        roundingArmsDiffered = 1;
  }

  /* A pinned export rounding would make the two Q15 arms identical on every
   * input; the corpus above must separate them at least once. */
  if (!roundingArmsDiffered) {
    printf("matmul_q31: the two declared Q15 roundings never disagreed\n");
    ++failures;
  }

  if (failures) {
    printf("matmul_q31: %d failures\n", failures);
    return 1;
  }
  return 0;
}
