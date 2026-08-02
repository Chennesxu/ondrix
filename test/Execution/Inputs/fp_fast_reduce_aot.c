#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for the fast-contract f32 reduction. The contract declares a
 * relaxation, so no trial pins a relaxed result to a bit pattern; the envelope
 * trials carry the error bound and the determinism check, and the directed
 * corpora additionally carry an executed divergence. Two families run without
 * the envelope: the integer-lattice corpus, whose sub-domain admits only exact
 * schedules, and the special-value corpus, where a finite bound is
 * meaningless. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32;

#define MAKE_MEMREF(DATA, COUNT)                                                                   \
  {                                                                                                \
    DATA, DATA, 0, {COUNT}, { 1 }                                                                  \
  }

extern float _mlir_ciface_f32_dot_fast(MemRefF32 *, MemRefF32 *);

enum { kMaxLength = 129 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* The kernel reads through descriptors, so each call gets its own copies and
 * cannot observe state left by the previous one. */
static float callKernel(const float *lhs, const float *rhs, int64_t count) {
  float lhsCopy[kMaxLength];
  float rhsCopy[kMaxLength];
  memcpy(lhsCopy, lhs, (size_t)count * sizeof(float));
  memcpy(rhsCopy, rhs, (size_t)count * sizeof(float));

  MemRefF32 lhsRef = MAKE_MEMREF(lhsCopy, count);
  MemRefF32 rhsRef = MAKE_MEMREF(rhsCopy, count);
  return _mlir_ciface_f32_dot_fast(&lhsRef, &rhsRef);
}

/* The repeated pair every family interprets: the value is usable only once the
 * two calls agree bit for bit. */
static int callTwice(const char *name, const float *lhs, const float *rhs, int64_t count,
                     float *result) {
  const float first = callKernel(lhs, rhs, count);
  const float second = callKernel(lhs, rhs, count);
  *result = first;
  if (memcmp(&first, &second, sizeof(first)) != 0) {
    fprintf(stderr, "%s (N=%lld): repeated calls returned %a and %a\n", name, (long long)count,
            (double)first, (double)second);
    return 1;
  }
  return 0;
}

/* The ordered scalar fma chain the fast contract is permitted to leave: one
 * fused event per element in increasing order, then the initial value. */
static float referenceOrderedFma(const float *lhs, const float *rhs, int64_t count) {
  const float initial = 0.0f;
  float accumulator = 0.0f;
  for (int64_t index = 0; index < count; ++index)
    accumulator = fmaf(lhs[index], rhs[index], accumulator);
  return accumulator + initial;
}

/* The higher-precision reference, plus the sum of absolute products the
 * envelope is stated relative to. */
static double referenceDouble(const float *lhs, const float *rhs, int64_t count,
                              double *absoluteSum) {
  double sum = 0.0;
  double magnitude = 0.0;
  for (int64_t index = 0; index < count; ++index) {
    const double product = (double)lhs[index] * (double)rhs[index];
    sum += product;
    magnitude += fabs(product);
  }
  *absoluteSum = magnitude;
  return sum;
}

/* One trial. `requireDivergence` marks the inputs directed at the fast/fma
 * split; a random trial may or may not diverge, so asserting it there would be
 * asserting a coincidence. Divergence is reproducible on the directed corpora
 * because the pinned LLVM 17.0.6 baseline fixes the emitted vector schedule. */
static int checkTrial(const char *name, const float *lhs, const float *rhs, int64_t count,
                      int requireDivergence) {
  float first = 0.0f;
  int failed = callTwice(name, lhs, rhs, count, &first);

  double absoluteSum = 0.0;
  const double expected = referenceDouble(lhs, rhs, count, &absoluteSum);
  /* 2^-24 is the f32 unit roundoff and 4*N bounds the rounding events any
   * regrouping of N products can accumulate; the floor keeps an all-zero
   * corpus from demanding an exact result for a relaxed contract. */
  const double tolerance = 4.0 * (double)count * 0x1p-24 * absoluteSum + 1e-30;
  const double error = fabs((double)first - expected);
  if (!(error <= tolerance)) {
    fprintf(stderr, "%s (N=%lld): got %a, reference %a, error %a exceeds %a\n", name,
            (long long)count, (double)first, expected, error, tolerance);
    failed = 1;
  }

  if (requireDivergence) {
    const float ordered = referenceOrderedFma(lhs, rhs, count);
    if (floatBits(first) == floatBits(ordered)) {
      fprintf(stderr,
              "%s (N=%lld): fast result %a matches the ordered fma chain, so the corpus "
              "does not separate the two schedules\n",
              name, (long long)count, (double)first);
      failed = 1;
    }
  }
  return failed;
}

/* (a) The unfused-product split, at length 12. Element 0 leaves the
 * accumulator at exactly 1.0 and element 1 has exact product
 * 2^-24 + 2^-48 - 2^-71: rounded on its own it lands on the 2^-24 tie that the
 * following add resolves back down, while the ordered chain fuses it and
 * rounds up. Observed fast 0x1p+0 against ordered 0x1.000002p+0. */
static int checkUnfusedProductCorpus(void) {
  float lhs[12];
  float rhs[12];

  for (int64_t index = 0; index < 12; ++index) {
    lhs[index] = 0.0f;
    rhs[index] = 0.0f;
  }
  lhs[0] = 1.0f;
  rhs[0] = 1.0f;
  lhs[1] = 0x1.000002p+0f;
  rhs[1] = 0x1.fffffep-25f;
  return checkTrial("unfused product", lhs, rhs, 12, 1);
}

/* (b) The cross-lane regrouping split, at length 20: two full vector passes
 * plus a four-element tail. The ordered chain absorbs elements 1..7 into a
 * 10^8 accumulator and loses them; the relaxed schedule cancels 10^8 against
 * element 8 inside lane 0 and keeps every unit term. Observed fast 0x1.2p+4
 * (18, the exact sum) against ordered 0x1.6p+3 (11). */
static int checkCrossLaneCorpus(void) {
  float lhs[20];
  float rhs[20];

  for (int64_t index = 0; index < 20; ++index) {
    lhs[index] = 1.0f;
    rhs[index] = 1.0f;
  }
  lhs[0] = 1.0e8f;
  lhs[8] = -1.0e8f;
  return checkTrial("cross-lane cancellation", lhs, rhs, 20, 1);
}

/* Lengths around the eight-lane boundary: pure tail, pure vector body, one
 * element past it, an exact multiple, and a long run with a tail. */
static int checkBoundaryLengths(void) {
  static const int64_t lengths[] = {7, 8, 9, 24, 100};
  float lhs[kMaxLength];
  float rhs[kMaxLength];
  int failed = 0;

  for (int64_t index = 0; index < kMaxLength; ++index) {
    lhs[index] = (index % 2 == 0 ? 1.0f : -1.0f) * (0.5f + 0.125f * (float)(index % 9));
    rhs[index] = 0.75f - 0.0625f * (float)(index % 11);
  }
  for (size_t entry = 0; entry < sizeof(lengths) / sizeof(lengths[0]); ++entry)
    failed |= checkTrial("boundary length", lhs, rhs, lengths[entry], 0);
  return failed;
}

/* Integer lattice. Both operands hold small integers, so every product is an
 * integer and the widest corpus keeps |product| <= 4096 over at most 129
 * elements: |partial sum| <= 129 * 4096 = 528384 < 2^23, an order of magnitude
 * inside the 2^24 exact-integer range of f32. No schedule derivable from the
 * reduction rounds anywhere in this sub-domain, so every lane assignment,
 * every fused or unfused product, and the cross-lane fold agree on the same
 * integer. Asserting that integer bit for bit therefore pins term conservation
 * and index coverage, not the relaxation: an envelope wide enough for a legal
 * regrouping also admits a dropped or duplicated product, and this assertion
 * does not. */
static int checkLatticeTrial(const char *name, const float *lhs, const float *rhs, int64_t count) {
  float result = 0.0f;
  int failed = callTwice(name, lhs, rhs, count, &result);

  int64_t exact = 0;
  for (int64_t index = 0; index < count; ++index)
    exact += (int64_t)lhs[index] * (int64_t)rhs[index];
  if (floatBits(result) != floatBits((float)exact)) {
    fprintf(stderr, "%s (N=%lld): got %a, exact integer sum %lld is %a\n", name, (long long)count,
            (double)result, (long long)exact, (double)(float)exact);
    failed = 1;
  }
  return failed;
}

/* Every product strictly positive, so dropping or repeating any single one
 * moves the sum. */
static void fillLatticePositive(float *lhs, float *rhs, int64_t count) {
  for (int64_t index = 0; index < count; ++index) {
    lhs[index] = (float)(1 + (int)(index % 9));
    rhs[index] = (float)(2 + (int)(index % 7));
  }
}

static void fillLatticeMixed(float *lhs, float *rhs, int64_t count) {
  for (int64_t index = 0; index < count; ++index) {
    lhs[index] = (float)((index % 5 == 0 ? -1 : 1) * (3 + (int)(index % 11)));
    rhs[index] = (float)((index % 3 == 0 ? -1 : 1) * (2 + (int)(index % 9)));
  }
}

/* The margin case: |product| between 3968 and 4096, so a lane running the full
 * 129 elements still accumulates far below the exact range. */
static void fillLatticeWide(float *lhs, float *rhs, int64_t count) {
  for (int64_t index = 0; index < count; ++index) {
    lhs[index] = (float)(64 - (int)(index % 3));
    rhs[index] = (float)(index % 4 == 0 ? -64 : 64);
  }
}

static int checkIntegerLatticeCorpus(void) {
  /* Pure tail, one full pass, a pass plus one, a pass plus a seven-element
   * tail, two full passes, three, eight, and sixteen with a one-element
   * tail. */
  static const int64_t lengths[] = {7, 8, 9, 15, 16, 24, 64, 129};
  float lhs[kMaxLength];
  float rhs[kMaxLength];
  int failed = 0;

  for (size_t entry = 0; entry < sizeof(lengths) / sizeof(lengths[0]); ++entry) {
    const int64_t count = lengths[entry];
    fillLatticePositive(lhs, rhs, count);
    failed |= checkLatticeTrial("lattice positive", lhs, rhs, count);
    fillLatticeMixed(lhs, rhs, count);
    failed |= checkLatticeTrial("lattice mixed sign", lhs, rhs, count);
    fillLatticeWide(lhs, rhs, count);
    failed |= checkLatticeTrial("lattice wide", lhs, rhs, count);
  }
  return failed;
}

enum SpecialClass { kClassZero, kClassPositiveInfinity, kClassNegativeInfinity, kClassNotANumber };

static const char *specialClassName(enum SpecialClass expected) {
  switch (expected) {
  case kClassZero:
    return "zero";
  case kClassPositiveInfinity:
    return "+inf";
  case kClassNegativeInfinity:
    return "-inf";
  case kClassNotANumber:
    return "nan";
  }
  return "unknown";
}

/* The envelope is a distance to a finite f64 reference and says nothing once a
 * product is non-finite, so these trials assert only the IEEE class the
 * schedule must carry through the lanes, the fold, and the tail. */
static int checkSpecialTrial(const char *name, const float *lhs, const float *rhs, int64_t count,
                             enum SpecialClass expected) {
  float result = 0.0f;
  int failed = callTwice(name, lhs, rhs, count, &result);
  int matched = 0;

  switch (expected) {
  case kClassZero:
    matched = result == 0.0f;
    break;
  case kClassPositiveInfinity:
    matched = isinf(result) && !signbit(result);
    break;
  case kClassNegativeInfinity:
    matched = isinf(result) && signbit(result);
    break;
  case kClassNotANumber:
    matched = isnan(result) != 0;
    break;
  }
  if (!matched) {
    fprintf(stderr, "%s (N=%lld): got %a, expected class %s\n", name, (long long)count,
            (double)result, specialClassName(expected));
    failed = 1;
  }
  return failed;
}

enum { kSpecialLength = 20, kSpecialBodyIndex = 3, kSpecialTailIndex = 17 };

static void fillUnitProducts(float *lhs, float *rhs, int64_t count) {
  for (int64_t index = 0; index < count; ++index) {
    lhs[index] = 1.0f;
    rhs[index] = 1.0f;
  }
}

/* Length 20 runs two full vector passes and a four-element tail, so index 3
 * carries the special value through a lane and the cross-lane fold while
 * index 17 carries it through the scalar tail. */
static int checkSpecialValueCorpus(void) {
  float lhs[kSpecialLength];
  float rhs[kSpecialLength];
  int failed = 0;

  for (int64_t index = 0; index < kSpecialLength; ++index) {
    lhs[index] = 0.0f;
    rhs[index] = 1.0f;
  }
  /* Observed 0x0p+0 for both zero trials: the dense<0.0> seed, the +0.0
   * reduction start, and the +0.0 initial keep a positive zero available at
   * every add, and IEEE addition returns +0 unless both operands are -0.
   * Either sign satisfies the assertion. */
  failed |= checkSpecialTrial("zero products", lhs, rhs, kSpecialLength, kClassZero);

  for (int64_t index = 0; index < kSpecialLength; ++index)
    rhs[index] = index % 2 == 0 ? 1.0f : -1.0f;
  failed |= checkSpecialTrial("signed zero products", lhs, rhs, kSpecialLength, kClassZero);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialBodyIndex] = INFINITY;
  failed |= checkSpecialTrial("positive infinity in body", lhs, rhs, kSpecialLength,
                              kClassPositiveInfinity);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialTailIndex] = INFINITY;
  failed |= checkSpecialTrial("positive infinity in tail", lhs, rhs, kSpecialLength,
                              kClassPositiveInfinity);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialBodyIndex] = -INFINITY;
  failed |= checkSpecialTrial("negative infinity in body", lhs, rhs, kSpecialLength,
                              kClassNegativeInfinity);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialTailIndex] = -INFINITY;
  failed |= checkSpecialTrial("negative infinity in tail", lhs, rhs, kSpecialLength,
                              kClassNegativeInfinity);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialBodyIndex] = INFINITY;
  lhs[kSpecialBodyIndex + 1] = -INFINITY;
  failed |=
      checkSpecialTrial("opposite infinities in body", lhs, rhs, kSpecialLength, kClassNotANumber);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialBodyIndex] = INFINITY;
  lhs[kSpecialTailIndex] = -INFINITY;
  failed |= checkSpecialTrial("opposite infinities across the tail", lhs, rhs, kSpecialLength,
                              kClassNotANumber);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialBodyIndex] = NAN;
  failed |= checkSpecialTrial("quiet nan in body", lhs, rhs, kSpecialLength, kClassNotANumber);

  fillUnitProducts(lhs, rhs, kSpecialLength);
  lhs[kSpecialTailIndex] = NAN;
  failed |= checkSpecialTrial("quiet nan in tail", lhs, rhs, kSpecialLength, kClassNotANumber);
  return failed;
}

static uint32_t nextRandom(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

static float randomValue(uint32_t *state) {
  const int16_t raw = (int16_t)(nextRandom(state) >> 16);
  return (float)raw / 16384.0f;
}

static int checkRandomCorpus(void) {
  static const int64_t lengths[] = {8, 12, 64, 129};
  float lhs[kMaxLength];
  float rhs[kMaxLength];
  uint32_t state = UINT32_C(0x9e3779b9);
  int failed = 0;

  for (int trial = 0; trial < 32; ++trial) {
    const int64_t count = lengths[trial % 4];
    char name[32];
    snprintf(name, sizeof(name), "random %d", trial);
    for (int64_t index = 0; index < count; ++index) {
      lhs[index] = randomValue(&state);
      rhs[index] = randomValue(&state);
    }
    failed |= checkTrial(name, lhs, rhs, count, 0);
  }
  return failed;
}

int main(void) {
  int failed = 0;
  failed |= checkUnfusedProductCorpus();
  failed |= checkCrossLaneCorpus();
  failed |= checkBoundaryLengths();
  failed |= checkIntegerLatticeCorpus();
  failed |= checkSpecialValueCorpus();
  failed |= checkRandomCorpus();
  return failed;
}
