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

extern void _mlir_ciface_rms64_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_rms2_q15(MemRefI16 *, MemRefI16 *);

enum { kExtent = 64, kTrialCount = 16 };

/* Independent contract arithmetic: exact sum of squares, nearest-even
 * mean by log2(N) in explicit floor-division form, then an exact floor
 * root from a corrected libm estimate (a different algorithm than the
 * compiled bit-by-bit root), nearest adjustment, and saturation. */
static int16_t referenceRms(const int16_t *input, int64_t extent) {
  int64_t sumsq = 0;
  for (int64_t i = 0; i < extent; ++i)
    sumsq += (int64_t)input[i] * input[i];
  int64_t quotient = sumsq / extent;
  int64_t remainder = sumsq % extent;
  int64_t half = extent / 2;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  int64_t root = (int64_t)sqrt((double)quotient);
  while (root > 0 && root * root > quotient)
    --root;
  while ((root + 1) * (root + 1) <= quotient)
    ++root;
  if (quotient - root * root > root)
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

static int check(void (*kernel)(MemRefI16 *, MemRefI16 *), const int16_t *input, int64_t extent,
                 const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {extent}, {1}};
  MemRefI16 output;
  kernel(&output, &inputRef);

  int16_t expected = referenceRms(input, extent);
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

int main(void) {
  int failed = 0;
  int16_t input[kExtent];

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 0;
  failed |= check(_mlir_ciface_rms64_q15, input, kExtent, "zero");

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = INT16_MAX;
  failed |= check(_mlir_ciface_rms64_q15, input, kExtent, "all max");

  /* All-minimum: mean is exactly 2^30 and the root saturates to 32767 —
   * the only reachable saturation of the contract. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = INT16_MIN;
  failed |= check(_mlir_ciface_rms64_q15, input, kExtent, "all min saturates");

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
  failed |= check(_mlir_ciface_rms64_q15, input, kExtent, "alternating");

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 0;
  input[0] = INT16_MAX;
  failed |= check(_mlir_ciface_rms64_q15, input, kExtent, "single impulse");

  /* Constant DC: mean of squares is the exact square 8192^2. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 8192;
  failed |= check(_mlir_ciface_rms64_q15, input, kExtent, "exact-square dc");

  uint32_t state = 0x0A15A75Cu;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    for (int64_t i = 0; i < kExtent; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "rms64 trial %d", trial);
    failed |= check(_mlir_ciface_rms64_q15, input, kExtent, label);
    snprintf(label, sizeof label, "rms2 trial %d", trial);
    failed |= check(_mlir_ciface_rms2_q15, input, 2, label);
  }
  return failed;
}
