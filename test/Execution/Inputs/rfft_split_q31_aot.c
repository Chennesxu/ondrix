/* Independent reference and execution gate for the half-size packed-Q31
 * split contract, written from the frozen equations with explicit wrapping,
 * floor, and toward-zero rounding. The gate refuses to pass unless the corpus
 * reaches a negative-odd value at both an i32 halving and an i64 combine
 * halving, the only places toward-zero and floor disagree, and it re-derives
 * both range claims: the ar clamp never fires and no combine reaches 2^32. */

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI64;

extern void _mlir_ciface_rfft_split8_q31(MemRefI64 *result, MemRefI64 *input);
extern void _mlir_ciface_rfft_split16_q31(MemRefI64 *result, MemRefI64 *input);

enum { kMaxExtent = 16, kTableExtent = 32 };

static int64_t twiddleSine[kTableExtent / 2 + 1];
static int64_t twiddleCosine[kTableExtent / 2 + 1];

static void buildTwiddles(void) {
  for (int k = 0; k <= kTableExtent / 2; ++k) {
    double angle = M_PI * (double)k / (double)kTableExtent;
    twiddleSine[k] = (int64_t)llround(sin(angle) * 1073741824.0);
    twiddleCosine[k] = (int64_t)llround(cos(angle) * 1073741824.0);
  }
}

static long narrowHalvings = 0;
static long combineHalvings = 0;
static long clampsFired = 0;
static int64_t worstCombine = 0;

/* Two's-complement wrap of a 33-bit intermediate into the i32 carrier. */
static int32_t wrap32(int64_t value) {
  uint32_t bits = (uint32_t)((uint64_t)value & 0xFFFFFFFFu);
  return bits <= (uint32_t)INT32_MAX ? (int32_t)bits
                                     : (int32_t)((int64_t)bits - (INT64_C(1) << 32));
}

static int64_t halveTowardZero(int64_t value, long *discriminating) {
  if (value < 0 && value % 2 != 0)
    ++*discriminating;
  return value / 2;
}

static int64_t halveFloor(int64_t value) {
  int64_t quotient = value / 2;
  if (value % 2 != 0 && value < 0)
    --quotient;
  return quotient;
}

static int64_t saturate32(int64_t value) {
  if (value > INT32_MAX) {
    ++clampsFired;
    return INT32_MAX;
  }
  if (value < INT32_MIN) {
    ++clampsFired;
    return INT32_MIN;
  }
  return value;
}

/* Toward-zero scaling of the exact i64 product by 2^30. */
static int64_t scaleProduct(int64_t coefficient, int64_t value) {
  return (coefficient * value) / (INT64_C(1) << 30);
}

static int64_t combine(int64_t value) {
  int64_t magnitude = value < 0 ? -value : value;
  if (magnitude > worstCombine)
    worstCombine = magnitude;
  return halveTowardZero(value, &combineHalvings);
}

static int32_t realOf(int64_t packed) { return wrap32(packed); }

static int32_t imagOf(int64_t packed) { return wrap32((int64_t)((uint64_t)packed >> 32)); }

/* The declared low-32 store of both components. */
static int64_t packBin(int64_t real, int64_t imaginary) {
  uint32_t realBits = (uint32_t)((uint64_t)real & 0xFFFFFFFFu);
  uint32_t imaginaryBits = (uint32_t)((uint64_t)imaginary & 0xFFFFFFFFu);
  return (int64_t)(((uint64_t)imaginaryBits << 32) | (uint64_t)realBits);
}

static void referenceSplit(const int64_t *input, int extent, int64_t *bins) {
  int half = extent / 2;
  int stride = kTableExtent / extent;

  int64_t dc = wrap32((int64_t)realOf(input[0]) + imagOf(input[0]));
  bins[0] = packBin(halveTowardZero(dc, &narrowHalvings), 0);

  for (int k = 1; k < half; ++k) {
    int64_t sine = twiddleSine[k * stride];
    int64_t cosine = twiddleCosine[k * stride];
    int64_t xr = realOf(input[k]), xi = imagOf(input[k]);
    int64_t mr = realOf(input[extent - k]), mi = imagOf(input[extent - k]);
    int64_t ar = saturate32(halveFloor(xr - mr));
    int64_t sr = halveTowardZero(wrap32(xr + mr), &narrowHalvings);
    int64_t ai = halveTowardZero(wrap32(xi - mi), &narrowHalvings);
    int64_t si = halveTowardZero(wrap32(xi + mi), &narrowHalvings);
    int64_t p1 = scaleProduct(sine, ar);
    int64_t p2 = scaleProduct(cosine, si);
    int64_t p3 = scaleProduct(sine, si);
    int64_t p4 = scaleProduct(cosine, ar);
    bins[k] = packBin(combine(sr - p1 + p2), combine(ai - p3 - p4));
    bins[extent - k] = packBin(combine(sr + p1 - p2), combine(-ai - p3 - p4));
  }

  int64_t selfSine = twiddleSine[half * stride];
  int64_t selfCosine = twiddleCosine[half * stride];
  int64_t selfXr = realOf(input[half]), selfXi = imagOf(input[half]);
  int64_t selfSr = halveTowardZero(wrap32(selfXr + selfXr), &narrowHalvings);
  int64_t selfSi = halveTowardZero(wrap32(selfXi + selfXi), &narrowHalvings);
  int64_t selfP2 = scaleProduct(selfCosine, selfSi);
  int64_t selfP3 = scaleProduct(selfSine, selfSi);
  bins[half] = packBin(combine(selfSr + selfP2), combine(-selfP3));
}

static int failures = 0;

static void checkVector(const char *name, const int64_t *input, int extent) {
  int64_t expected[kMaxExtent];
  referenceSplit(input, extent, expected);

  MemRefI64 inputRef = {(int64_t *)input, (int64_t *)input, 0, {extent}, {1}};
  MemRefI64 output;
  if (extent == 8)
    _mlir_ciface_rfft_split8_q31(&output, &inputRef);
  else
    _mlir_ciface_rfft_split16_q31(&output, &inputRef);
  if (output.sizes[0] != extent) {
    printf("FAIL %s: expected extent %d, got %" PRId64 "\n", name, extent, output.sizes[0]);
    ++failures;
    free(output.allocated);
    return;
  }
  for (int k = 0; k < extent; ++k) {
    int64_t actual = output.aligned[output.offset + k * output.strides[0]];
    if (actual != expected[k]) {
      printf("FAIL %s: bin %d = %016" PRIx64 " but reference = %016" PRIx64 "\n", name, k,
             (uint64_t)actual, (uint64_t)expected[k]);
      ++failures;
    }
  }
  free(output.allocated);
}

static void fillFromComponents(int64_t *input, const int32_t *components, int extent) {
  for (int i = 0; i < extent; ++i)
    input[i] = packBin(components[2 * i], components[2 * i + 1]);
}

static uint64_t nextState(uint64_t state) {
  return state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
}

int main(void) {
  buildTwiddles();

  int64_t input[kMaxExtent];
  int32_t components[2 * kMaxExtent];
  char name[64];

  for (int extent = 8; extent <= kMaxExtent; extent *= 2) {
    for (int i = 0; i < extent; ++i)
      input[i] = 0;
    snprintf(name, sizeof(name), "zeros %d", extent);
    checkVector(name, input, extent);

    for (int i = 0; i < 2 * extent; ++i)
      components[i] = INT32_MAX;
    fillFromComponents(input, components, extent);
    snprintf(name, sizeof(name), "positive rail %d", extent);
    checkVector(name, input, extent);

    for (int i = 0; i < 2 * extent; ++i)
      components[i] = (i & 1) ? INT32_MIN : INT32_MAX;
    fillFromComponents(input, components, extent);
    snprintf(name, sizeof(name), "alternating rail %d", extent);
    checkVector(name, input, extent);

    /* A single non-zero component separates bin k from its mirror N-k. */
    for (int position = 0; position < 2 * extent; ++position) {
      for (int i = 0; i < 2 * extent; ++i)
        components[i] = 0;
      components[position] = INT32_MAX;
      fillFromComponents(input, components, extent);
      snprintf(name, sizeof(name), "positive impulse %d %d", extent, position);
      checkVector(name, input, extent);
      components[position] = INT32_MIN;
      fillFromComponents(input, components, extent);
      snprintf(name, sizeof(name), "negative impulse %d %d", extent, position);
      checkVector(name, input, extent);
    }

    uint64_t state = UINT64_C(0x20260820) + (uint64_t)extent;
    for (int trial = 0; trial < 64; ++trial) {
      for (int i = 0; i < 2 * extent; ++i) {
        state = nextState(state);
        components[i] = wrap32((int64_t)(uint32_t)(state >> 32));
      }
      fillFromComponents(input, components, extent);
      snprintf(name, sizeof(name), "congruential %d %d", extent, trial);
      checkVector(name, input, extent);
    }
  }

  if (narrowHalvings == 0 || combineHalvings == 0) {
    printf("rfft_split gate is vacuous: negative-odd halvings %ld narrow, %ld combine\n",
           narrowHalvings, combineHalvings);
    return 1;
  }
  if (clampsFired != 0) {
    printf("rfft_split contract violated: the ar clamp fired %ld times\n", clampsFired);
    return 1;
  }
  if (worstCombine >= (INT64_C(1) << 32)) {
    printf("rfft_split contract violated: combine magnitude %" PRId64 "\n", worstCombine);
    return 1;
  }
  if (failures != 0) {
    printf("rfft_split execution gate failed: %d\n", failures);
    return 1;
  }
  printf("rfft_split execution gate passed: %ld/%ld negative-odd halvings, worst combine %" PRId64
         "\n",
         narrowHalvings, combineHalvings, worstCombine);
  return 0;
}
