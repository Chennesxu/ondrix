#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Object gate for a genuinely wrapping i48 accumulator.
 *
 * Exact-modulo reassociation legality does not depend on a wrap actually
 * occurring: addition modulo 2^48 is associative and commutative, so the
 * horizontal Vector schedule and the ordered scalar schedule agree whether
 * or not the accumulator leaves [-2^47, 2^47). The committed evidence so
 * far only covered accumulators proven never to wrap, so this harness
 * drives inputs whose exact sum of products definitely exceeds 2^47 and
 * pins horizontal == ordered == independent reference. */

typedef int16_t (*kernel_t)(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                            int64_t, int64_t, int64_t);

extern int16_t q15_i48_wrap_horizontal(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                       int16_t *, int64_t, int64_t, int64_t);
extern int16_t q15_i48_wrap_ordered(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                    int16_t *, int64_t, int64_t, int64_t);
extern int64_t q15_i48_wrap_raw_horizontal(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                           int16_t *, int16_t *, int64_t, int64_t, int64_t);
extern int64_t q15_i48_wrap_raw_ordered(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                        int16_t *, int64_t, int64_t, int64_t);

enum { kMaxLength = 300100, kSparseStride = 3, kSparseOffset = 1 };

/* Independent reference: exact sum of products in __int128, reduced modulo
 * 2^48 and sign-folded into [-2^47, 2^47), then the declared Q15 export —
 * nearest-even shift by 15 in explicit floor-division form, saturating to
 * i16. `wrapped` reports whether any ORDERED PREFIX of the exact sum left
 * the i48 range, which is when the ordered schedule actually wraps; the
 * final sum can walk back inside afterwards. `raw` receives the sign-folded
 * i48 accumulator itself, which the identity i64/frac30 export materializes
 * unchanged. */
static int16_t reference(const int16_t *lhs, int64_t lhsOffset, int64_t lhsStride,
                         const int16_t *rhs, int64_t rhsOffset, int64_t rhsStride, int64_t length,
                         int *wrapped, int64_t *raw) {
  const __int128 half = (__int128)1 << 47;
  __int128 exact = 0;
  *wrapped = 0;
  for (int64_t i = 0; i < length; ++i) {
    exact += (__int128)lhs[lhsOffset + i * lhsStride] * (__int128)rhs[rhsOffset + i * rhsStride];
    if (exact < -half || exact >= half)
      *wrapped = 1;
  }

  const unsigned __int128 mask = ((unsigned __int128)1 << 48) - 1;
  const unsigned __int128 bits = (unsigned __int128)exact & mask;
  __int128 folded = (__int128)bits;
  if (bits >> 47)
    folded -= (__int128)1 << 48;
  *raw = (int64_t)folded;

  const __int128 divisor = (__int128)1 << 15;
  __int128 quotient = folded / divisor;
  __int128 remainder = folded % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  if (remainder > divisor / 2 || (remainder == divisor / 2 && (quotient & 1)))
    ++quotient;
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return (int16_t)quotient;
}

/* One logical operand pair materialized twice: densely for the identity
 * layout the Vector path requires, and at stride 3 with a nonzero offset
 * for the dynamic-layout kernel that stays ordered. */
static void materialize(const int16_t *lhs, const int16_t *rhs, int64_t length, int16_t *sparseLhs,
                        int16_t *sparseRhs) {
  for (int64_t i = 0; i < length * kSparseStride + kSparseOffset; ++i) {
    sparseLhs[i] = INT16_MAX;
    sparseRhs[i] = INT16_MIN;
  }
  for (int64_t i = 0; i < length; ++i) {
    sparseLhs[kSparseOffset + i * kSparseStride] = lhs[i];
    sparseRhs[kSparseOffset + i * kSparseStride] = rhs[i];
  }
}

/* `expectNonSaturating` marks the cases whose folded accumulator still fits
 * the Q15 destination. Those are the discriminating ones: an implementation
 * that saturated the accumulator instead of wrapping it would reach the
 * export clamp and be caught, while a case whose export saturates anyway
 * cannot distinguish the two policies on its own. */
static int checkCase(const char *name, int16_t *lhs, int16_t *rhs, int64_t length,
                     int16_t *sparseLhs, int16_t *sparseRhs, int expectWrap,
                     int expectNonSaturating) {
  int wrapped = 0;
  int64_t expectedRaw = 0;
  const int16_t expected = reference(lhs, 0, 1, rhs, 0, 1, length, &wrapped, &expectedRaw);
  materialize(lhs, rhs, length, sparseLhs, sparseRhs);

  const int16_t horizontal =
      q15_i48_wrap_horizontal(lhs, lhs, 0, length, 1, rhs, rhs, 0, length, 1);
  const int16_t ordered =
      q15_i48_wrap_ordered(sparseLhs, sparseLhs, kSparseOffset, length, kSparseStride, sparseRhs,
                           sparseRhs, kSparseOffset, length, kSparseStride);
  const int64_t rawHorizontal =
      q15_i48_wrap_raw_horizontal(lhs, lhs, 0, length, 1, rhs, rhs, 0, length, 1);
  const int64_t rawOrdered =
      q15_i48_wrap_raw_ordered(sparseLhs, sparseLhs, kSparseOffset, length, kSparseStride,
                               sparseRhs, sparseRhs, kSparseOffset, length, kSparseStride);

  int failed = 0;
  if (horizontal != expected || ordered != expected) {
    fprintf(stderr, "%s length %lld: reference %d, horizontal %d, ordered %d\n", name,
            (long long)length, expected, horizontal, ordered);
    failed = 1;
  }
  /* The stronger equality: the folded i48 accumulator itself, before the
   * Q15 destination can hide a difference behind its clamp. */
  if (rawHorizontal != expectedRaw || rawOrdered != expectedRaw) {
    fprintf(stderr, "%s length %lld: raw reference %lld, horizontal %lld, ordered %lld\n", name,
            (long long)length, (long long)expectedRaw, (long long)rawHorizontal,
            (long long)rawOrdered);
    failed = 1;
  }
  if (expectWrap && !wrapped) {
    fprintf(stderr, "%s length %lld: the i48 accumulator did not actually wrap\n", name,
            (long long)length);
    failed = 1;
  }
  if (expectNonSaturating && (expected == 32767 || expected == -32768)) {
    fprintf(stderr, "%s length %lld: export saturated at %d, so the case proves nothing\n", name,
            (long long)length, expected);
    failed = 1;
  }
  return failed;
}

int main(void) {
  int16_t *lhs = malloc((size_t)kMaxLength * sizeof(int16_t));
  int16_t *rhs = malloc((size_t)kMaxLength * sizeof(int16_t));
  int16_t *sparseLhs =
      malloc(((size_t)kMaxLength * kSparseStride + kSparseOffset) * sizeof(int16_t));
  int16_t *sparseRhs =
      malloc(((size_t)kMaxLength * kSparseStride + kSparseOffset) * sizeof(int16_t));
  if (!lhs || !rhs || !sparseLhs || !sparseRhs) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }

  int failed = 0;

  /* (a) 200000 terms of INT16_MIN * INT16_MIN = +2^30. The exact sum
   * 200000 * 2^30 = 214748364800000 exceeds 2^47 = 140737488355328, so the
   * i48 accumulator folds into negative territory. The Q15 export of that
   * folded value is far outside the destination and clamps. */
  const int64_t allMinimumLength = 200000;
  for (int64_t i = 0; i < allMinimumLength; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MIN;
  }
  failed |= checkCase("all-minimum", lhs, rhs, allMinimumLength, sparseLhs, sparseRhs,
                      /*expectWrap=*/1, /*expectNonSaturating=*/0);

  /* (b) A mixed-sign pattern that crosses the 2^47 boundary mid-fold and
   * then walks back inside it: 150000 terms of +2^30 already exceed 2^47,
   * and the following 60000 terms of -32768 * 32767 bring the exact sum
   * back to 96638730240000, which is representable in i48. The ordered
   * schedule wraps and unwraps while the horizontal schedule visits
   * entirely different partial sums; both must still land on the same
   * exported value. */
  const int64_t positiveRun = 150000;
  const int64_t mixedLength = 210000;
  for (int64_t i = 0; i < positiveRun; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MIN;
  }
  for (int64_t i = positiveRun; i < mixedLength; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MAX;
  }
  failed |= checkCase("mixed-sign crossing", lhs, rhs, mixedLength, sparseLhs, sparseRhs,
                      /*expectWrap=*/1, /*expectNonSaturating=*/0);

  /* (a2) The same wrap made observable at the destination. 2^18 = 262144
   * terms of +2^30 are exactly 2^48, so the accumulator completes a full
   * turn back to zero; three small products then leave 3000000, whose
   * nearest-even shift by 15 is 92. A saturating accumulator would reach
   * the export clamp instead, so this case actually discriminates. */
  const int64_t fullTurn = 262144;
  const int64_t fullTurnLength = fullTurn + 3;
  for (int64_t i = 0; i < fullTurn; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MIN;
  }
  for (int64_t i = fullTurn; i < fullTurnLength; ++i) {
    lhs[i] = 1000;
    rhs[i] = 1000;
  }
  failed |= checkCase("full-turn", lhs, rhs, fullTurnLength, sparseLhs, sparseRhs,
                      /*expectWrap=*/1, /*expectNonSaturating=*/1);

  /* (b2) The mixed-sign crossing made observable in the same way. Each
   * (+2^30, -32768 * 32767) pair contributes exactly 32768, and four extra
   * negative terms bring 150000 positive terms down to 620363776 — inside
   * the destination range, exported as 18932. */
  const int64_t walkBackPositive = 150000;
  const int64_t walkBackLength = walkBackPositive + 150004;
  for (int64_t i = 0; i < walkBackPositive; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MIN;
  }
  for (int64_t i = walkBackPositive; i < walkBackLength; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MAX;
  }
  failed |= checkCase("walk-back", lhs, rhs, walkBackLength, sparseLhs, sparseRhs,
                      /*expectWrap=*/1, /*expectNonSaturating=*/1);

  /* (b3) The same crossing in the negative direction: 150004 terms of
   * -32768 * 32767 take the running value below -2^47, then 150000 terms of
   * +2^30 bring it back to 620363776 and three -1000 * 1000 products leave
   * 617363776, exported as 18840. */
  const int64_t negativeRun = 150004;
  const int64_t negativeReturn = negativeRun + 150000;
  const int64_t negativeLength = negativeReturn + 3;
  for (int64_t i = 0; i < negativeRun; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MAX;
  }
  for (int64_t i = negativeRun; i < negativeReturn; ++i) {
    lhs[i] = INT16_MIN;
    rhs[i] = INT16_MIN;
  }
  for (int64_t i = negativeReturn; i < negativeLength; ++i) {
    lhs[i] = 1000;
    rhs[i] = -1000;
  }
  failed |= checkCase("negative crossing", lhs, rhs, negativeLength, sparseLhs, sparseRhs,
                      /*expectWrap=*/1, /*expectNonSaturating=*/1);

  /* (c) Deterministic pseudo-random trials. Each keeps the 2^48 full-turn
   * prefix so the accumulator provably wraps, then appends a random tail
   * whose products are small enough that the exported value stays inside
   * the destination and remains rounding sensitive. */
  uint32_t state = UINT32_C(0x9e3779b9);
  for (int trial = 0; trial < 8; ++trial) {
    const int64_t tail = 500 + trial * 100;
    const int64_t length = fullTurn + tail;
    for (int64_t i = 0; i < fullTurn; ++i) {
      lhs[i] = INT16_MIN;
      rhs[i] = INT16_MIN;
    }
    for (int64_t i = fullTurn; i < length; ++i) {
      state = state * UINT32_C(1664525) + UINT32_C(1013904223);
      lhs[i] = (int16_t)((int32_t)((state >> 16) % 2001) - 1000);
      state = state * UINT32_C(1664525) + UINT32_C(1013904223);
      rhs[i] = (int16_t)((int32_t)((state >> 16) % 2001) - 1000);
    }
    failed |= checkCase("random tail", lhs, rhs, length, sparseLhs, sparseRhs,
                        /*expectWrap=*/1, /*expectNonSaturating=*/1);
  }

  free(lhs);
  free(rhs);
  free(sparseLhs);
  free(sparseRhs);
  return failed;
}
