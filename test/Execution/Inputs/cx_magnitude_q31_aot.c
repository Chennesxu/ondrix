/* Independent packed-Q31 magnitude reference.
 *
 * It derives the component pre-shift from the width rather than taking it as a
 * constant, unpacks with explicit arithmetic rather than the lowering's shift
 * pair, and reaches the root through a corrected libm estimate instead of the
 * compiled bit-by-bit search.
 */

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

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_magnitude_q31_even(MemRefI32 *, MemRefI64 *);
extern void _mlir_ciface_magnitude_q31_floor(MemRefI32 *, MemRefI64 *);

enum { kEven = 0, kFloor = 1, kExtent = 32 };

/* 2*(31 - k) + 1 <= 62 forces exactly one bit. */
static unsigned componentPreShift(void) {
  unsigned k = 0;
  while (2u * (31u - k) + 1u > 62u)
    ++k;
  return k;
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
  return (quotient & 1) ? quotient + 1 : quotient;
}

static int32_t reference(int64_t packed, int inputMode, int rootMode) {
  unsigned shift = componentPreShift();
  int32_t real = (int32_t)(uint32_t)((uint64_t)packed & 0xFFFFFFFFu);
  int32_t imaginary = (int32_t)(uint32_t)(((uint64_t)packed >> 32) & 0xFFFFFFFFu);
  int64_t re = roundShift(real, shift, inputMode);
  int64_t im = roundShift(imaginary, shift, inputMode);
  /* The restored sum reaches 2^63 at the full-scale corners, so the reference
   * carries it in 128 bits. Shifting it in int64_t is the undefined behavior
   * that let an earlier version of this gate agree with a broken lowering. */
  __int128 restored = ((__int128)(re * re + im * im)) << (2u * shift);
  __int128 root = (__int128)sqrt((double)(long double)restored);
  if (root < 0)
    root = 0;
  while (root > 0 && root * root > restored)
    --root;
  while ((root + 1) * (root + 1) <= restored)
    ++root;
  if (rootMode != kFloor && restored - root * root > root)
    ++root;
  if (root > INT32_MAX)
    return INT32_MAX;
  return (int32_t)root;
}

static int failures = 0;
static uint32_t state = 0x0fedcba9u;

static int32_t nextRandom(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (int32_t)state;
}

static int64_t pack(int32_t real, int32_t imaginary) {
  return (int64_t)(((uint64_t)(uint32_t)imaginary << 32) | (uint32_t)real);
}

static void check(const char *name, void (*kernel)(MemRefI32 *, MemRefI64 *), int64_t *packed,
                  int inputMode, int rootMode, int32_t *observedOut) {
  MemRefI64 in = {packed, packed, 0, {kExtent}, {1}};
  MemRefI32 out;
  kernel(&out, &in);
  for (int64_t i = 0; i < kExtent; ++i) {
    int32_t observed = out.aligned[out.offset + i * out.strides[0]];
    int32_t expected = reference(packed[i], inputMode, rootMode);
    observedOut[i] = observed;
    if (observed != expected) {
      printf("%s[%lld]: observed %d expected %d\n", name, (long long)i, observed, expected);
      ++failures;
    }
  }
  free(out.allocated);
}

int main(void) {
  static int64_t packed[kExtent];
  static int32_t even[kExtent], floorArm[kExtent];
  int armsDiffered = 0;

  /* The corner the boundary exists for, plus the other rails. */
  const int32_t rails[] = {INT32_MIN, INT32_MAX, 0, 1, -1, 1 << 30, -(1 << 30)};
  const unsigned railCount = sizeof(rails) / sizeof(rails[0]);
  for (unsigned a = 0; a < railCount; ++a)
    for (unsigned b = 0; b < railCount; ++b) {
      for (int64_t i = 0; i < kExtent; ++i)
        packed[i] = pack(rails[a], rails[b]);
      check("rail_even", _mlir_ciface_magnitude_q31_even, packed, kEven, kEven, even);
      check("rail_floor", _mlir_ciface_magnitude_q31_floor, packed, kFloor, kFloor, floorArm);
    }

  /* The saturating corner is named, not merely swept: its exact sum of squares
   * is 2^63 and an unclamped restore wraps it negative, which the square root
   * then reports as zero. */
  for (int64_t i = 0; i < kExtent; ++i)
    packed[i] = pack(INT32_MIN, INT32_MIN);
  check("corner_even", _mlir_ciface_magnitude_q31_even, packed, kEven, kEven, even);
  if (even[0] != INT32_MAX) {
    printf("corner (INT32_MIN, INT32_MIN): observed %d expected %d\n", even[0], INT32_MAX);
    ++failures;
  }

  for (int trial = 0; trial < 300; ++trial) {
    for (int64_t i = 0; i < kExtent; ++i)
      packed[i] = pack(nextRandom(), nextRandom());
    check("even", _mlir_ciface_magnitude_q31_even, packed, kEven, kEven, even);
    check("floor", _mlir_ciface_magnitude_q31_floor, packed, kFloor, kFloor, floorArm);
    for (int64_t i = 0; i < kExtent; ++i)
      if (even[i] != floorArm[i])
        armsDiffered = 1;
  }

  if (!armsDiffered) {
    printf("cx_magnitude_q31: the two declared roundings never disagreed\n");
    ++failures;
  }
  if (failures) {
    printf("cx_magnitude_q31: %d failures\n", failures);
    return 1;
  }
  return 0;
}
