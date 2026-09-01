/* Independent Q31 root-mean-square reference.
 *
 * It follows the declared boundaries -- the derived pre-shift, the
 * nearest-even mean, the restored root -- but reaches the root through a
 * corrected libm estimate rather than the compiled bit-by-bit search, and it
 * derives the pre-shift from the extent instead of taking it as a constant.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_rms16_q31(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_rms64_q31(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_rms64_q31_floor(MemRefI32 *, MemRefI32 *);

/* The smallest shift that keeps 2^m squares of a signed Q1.31 sample exact
 * in i64: 2*(31 - k) + m <= 62. */
static unsigned preShiftFor(int64_t extent) {
  unsigned m = 0;
  while ((int64_t)1 << (m + 1) <= extent)
    ++m;
  unsigned exact = 62 + m;
  return exact <= 62 ? 0 : (exact - 62 + 1) / 2;
}

/* One right shift with the declared tie rule; a right shift of an i32 never
 * leaves the destination, so the declared saturation cannot fire. */
static int64_t shiftRound(int32_t value, unsigned shift, int nearest) {
  int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = (int64_t)value / divisor;
  int64_t remainder = (int64_t)value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  if (!nearest)
    return quotient;
  int64_t half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return quotient;
}

static int32_t referenceRms(const int32_t *input, int64_t extent, int nearestInput,
                            int nearestRoot) {
  unsigned shift = preShiftFor(extent);
  int64_t sumsq = 0;
  for (int64_t i = 0; i < extent; ++i) {
    int64_t narrowed = shiftRound(input[i], shift, nearestInput);
    sumsq += narrowed * narrowed;
  }
  /* Nearest-even mean by the extent, which is a power of two. */
  int64_t quotient = sumsq / extent;
  int64_t remainder = sumsq % extent;
  int64_t half = extent / 2;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  /* Restore the 2k before the root, exactly as the contract states. */
  int64_t restored = quotient << (2 * shift);
  int64_t root = (int64_t)sqrt((double)restored);
  while (root > 0 && root * root > restored)
    --root;
  while ((root + 1) * (root + 1) <= restored)
    ++root;
  if (nearestRoot && restored - root * root > root)
    ++root;
  if (root > INT32_MAX)
    return INT32_MAX;
  return (int32_t)root;
}

static int failures = 0;

static void check(const char *name, void (*kernel)(MemRefI32 *, MemRefI32 *), const int32_t *input,
                  int64_t extent, int nearestInput, int nearestRoot, const char *caseName) {
  MemRefI32 in = {(int32_t *)input, (int32_t *)input, 0, {extent}, {1}};
  MemRefI32 out;
  kernel(&out, &in);
  int32_t observed = out.aligned[out.offset];
  int32_t expected = referenceRms(input, extent, nearestInput, nearestRoot);
  if (observed != expected) {
    printf("%s/%s: observed %d expected %d\n", name, caseName, observed, expected);
    ++failures;
  }
  free(out.allocated);
}

extern void free(void *);

int main(void) {
  static int32_t buffer[64];

  struct {
    const char *name;
    int32_t fill;
  } levels[] = {{"zero", 0}, {"max", INT32_MAX}, {"min", INT32_MIN},   {"half", 1 << 30},
                {"one", 1},  {"neg_one", -1},    {"dc_2p20", 1 << 20}, {"odd", 1234567}};

  for (unsigned level = 0; level < sizeof(levels) / sizeof(levels[0]); ++level) {
    for (int64_t i = 0; i < 64; ++i)
      buffer[i] = levels[level].fill;
    check("rms16", _mlir_ciface_rms16_q31, buffer, 16, 1, 1, levels[level].name);
    check("rms64", _mlir_ciface_rms64_q31, buffer, 64, 1, 1, levels[level].name);
    check("rms64_floor", _mlir_ciface_rms64_q31_floor, buffer, 64, 0, 0, levels[level].name);
  }

  /* The corner that moves the root across the clamp: sixty-three minimum
   * samples and one maximum. */
  for (int64_t i = 0; i < 64; ++i)
    buffer[i] = INT32_MIN;
  buffer[17] = INT32_MAX;
  check("rms64", _mlir_ciface_rms64_q31, buffer, 64, 1, 1, "mixed_rail");
  check("rms64_floor", _mlir_ciface_rms64_q31_floor, buffer, 64, 0, 0, "mixed_rail");

  /* A single full-scale impulse, where the mean is far below the rail. */
  for (int64_t i = 0; i < 64; ++i)
    buffer[i] = 0;
  buffer[3] = INT32_MIN;
  check("rms16", _mlir_ciface_rms16_q31, buffer, 16, 1, 1, "impulse");
  check("rms64", _mlir_ciface_rms64_q31, buffer, 64, 1, 1, "impulse");

  /* A xorshift corpus over the whole domain. */
  uint32_t state = 0x1234567u;
  for (int trial = 0; trial < 256; ++trial) {
    for (int64_t i = 0; i < 64; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      buffer[i] = (int32_t)state;
    }
    check("rms16", _mlir_ciface_rms16_q31, buffer, 16, 1, 1, "xorshift");
    check("rms64", _mlir_ciface_rms64_q31, buffer, 64, 1, 1, "xorshift");
    check("rms64_floor", _mlir_ciface_rms64_q31_floor, buffer, 64, 0, 0, "xorshift");
  }

  if (failures) {
    printf("rms_q31: %d failures\n", failures);
    return 1;
  }
  return 0;
}
