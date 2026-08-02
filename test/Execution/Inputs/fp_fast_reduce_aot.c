#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for the fast-contract f32 reduction. The contract declares a
 * relaxation, so nothing here compares the kernel against a pinned bit
 * pattern; every trial carries the envelope and the determinism check, and the
 * directed corpora additionally carry an executed divergence. */

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
  const float first = callKernel(lhs, rhs, count);
  const float second = callKernel(lhs, rhs, count);
  int failed = 0;

  if (memcmp(&first, &second, sizeof(first)) != 0) {
    fprintf(stderr, "%s (N=%lld): repeated calls returned %a and %a\n", name, (long long)count,
            (double)first, (double)second);
    failed = 1;
  }

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
  failed |= checkRandomCorpus();
  return failed;
}
