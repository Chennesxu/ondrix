#include <math.h>
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

extern void _mlir_ciface_q15_rms(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_q15_rms_floor(MemRefI16 *, MemRefI16 *);

enum { kExtent = 64, kTrialCount = 16 };

/* Independent contract arithmetic: exact sum of squares, the nearest-even
 * mean by the extent in explicit floor-division form, then an exact floor
 * root from a corrected libm estimate (a different algorithm than the
 * compiled bit-by-bit root), the declared root adjustment, and saturation.
 * Identical to the reference that pins rms_q15_vector_aot.mlir; here it
 * decides the frontend-compiled kernels. */
static int16_t referenceRms(const int16_t *input, int nearest) {
  int64_t sumsq = 0;
  for (int64_t i = 0; i < kExtent; ++i)
    sumsq += (int64_t)input[i] * input[i];
  int64_t quotient = sumsq / kExtent;
  int64_t remainder = sumsq % kExtent;
  int64_t half = kExtent / 2;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  int64_t root = (int64_t)sqrt((double)quotient);
  while (root > 0 && root * root > quotient)
    --root;
  while ((root + 1) * (root + 1) <= quotient)
    ++root;
  if (nearest && quotient - root * root > root)
    ++root;
  if (root > 32767)
    root = 32767;
  return (int16_t)root;
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int16_t toSigned16(uint32_t bits) {
  uint32_t low = bits & 0xFFFFu;
  return (int16_t)(low < 32768u ? (int32_t)low : (int32_t)low - 65536);
}

static int check(void (*kernel)(MemRefI16 *, MemRefI16 *), const int16_t *input, int nearest,
                 const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {kExtent}, {1}};
  MemRefI16 output;
  kernel(&output, &inputRef);

  int16_t expected = referenceRms(input, nearest);
  int failed = output.sizes[0] != 1;
  if (!failed) {
    int16_t actual = output.aligned[output.offset];
    if (actual != expected) {
      fprintf(stderr, "%s: got %d, expected %d\n", label, actual, expected);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

static int checkBoth(const int16_t *input, const char *label) {
  char sub[48];
  int failed = 0;
  snprintf(sub, sizeof sub, "%s nearest", label);
  failed |= check(_mlir_ciface_q15_rms, input, 1, sub);
  snprintf(sub, sizeof sub, "%s floor", label);
  failed |= check(_mlir_ciface_q15_rms_floor, input, 0, sub);
  return failed;
}

int main(void) {
  int failed = 0;
  int16_t input[kExtent];

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 0;
  failed |= checkBoth(input, "zero");

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = INT16_MAX;
  failed |= checkBoth(input, "all max");

  /* All-minimum: mean is exactly 2^30 and the root saturates to 32767. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = INT16_MIN;
  failed |= checkBoth(input, "all min saturates");

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
  failed |= checkBoth(input, "alternating");

  /* Constant DC: mean of squares is the exact square 8192^2, so the two
   * declared root roundings must agree here. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 8192;
  failed |= checkBoth(input, "exact-square dc");

  /* Divergence witness for the source-level root_rounding parameter: 32
   * zeros and 32 eights give mean 32, floor root 5 but nearest root 6. The
   * guard asserts the references themselves disagree so this case cannot
   * silently rot into one that no longer separates the two kernels. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = (i < 32) ? 0 : 8;
  if (referenceRms(input, 1) == referenceRms(input, 0)) {
    fprintf(stderr, "divergence witness no longer separates the root roundings\n");
    failed = 1;
  }
  failed |= checkBoth(input, "rounding divergence");

  uint32_t state = 0x51E6B3A7u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    for (int64_t i = 0; i < kExtent; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "trial %d", trial);
    failed |= checkBoth(input, label);
  }
  return failed;
}
