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

extern void _mlir_ciface_rms64_q15_vector(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_rms64_floor_q15_vector(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_rms4096_q15_vector(MemRefI16 *, MemRefI16 *);

enum { kExtent = 64, kMaxExtent = 4096, kTrialCount = 16 };

/* Independent contract arithmetic: exact sum of squares, then the mean by
 * log2(N) in explicit floor-division form, then an exact floor root from a
 * corrected libm estimate (a different algorithm than the compiled bit-by-bit
 * root), the nearest adjustment, and saturation. The sum of squares is never
 * negative, so the floor quotient needs no sign correction; the compiled
 * kernel reaches the same values through a horizontal Vector reduction over
 * an i64 wrapping accumulator and one acc_export at frac 30 - log2(N). */
static int16_t referenceRms(const int16_t *input, int64_t extent, int nearest) {
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

static int check(void (*kernel)(MemRefI16 *, MemRefI16 *), const int16_t *input, int64_t extent,
                 int nearest, const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {extent}, {1}};
  MemRefI16 output;
  kernel(&output, &inputRef);

  int16_t expected = referenceRms(input, extent, nearest);
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
  static int16_t input[kMaxExtent];

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 0;
  failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, "zero");

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = INT16_MAX;
  failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, "all max");

  /* All-minimum: mean is exactly 2^30 and the root saturates to 32767. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = INT16_MIN;
  failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, "all min saturates");

  /* Saturation is not unique to the all-minimum corner: sixty-three minimum
   * samples and one maximum leave the mean just below 2^30, but the nearest
   * adjustment lifts the floor root 32767 to 32768 before the clamp. This is
   * the witness the scalar gate pins; the Vector path must reproduce it even
   * though its lane order differs. */
  input[0] = INT16_MAX;
  failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, "mixed saturates");
  failed |= check(_mlir_ciface_rms64_floor_q15_vector, input, kExtent, 0, "mixed saturates floor");

  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
  failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, "alternating");

  /* One full-scale sample: the tail of the reduction is all zero, so the
   * whole result rides on a single lane of one vector chunk. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 0;
  input[0] = INT16_MAX;
  failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, "single impulse");

  /* Constant DC: mean of squares is the exact square 8192^2. */
  for (int64_t i = 0; i < kExtent; ++i)
    input[i] = 8192;
  failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, "exact-square dc");

  /* Largest admitted extent at the extreme magnitude: 4096 * 2^30 = 2^42,
   * the bound that keeps the i64 wrapping accumulator from wrapping. */
  for (int64_t i = 0; i < kMaxExtent; ++i)
    input[i] = INT16_MIN;
  failed |= check(_mlir_ciface_rms4096_q15_vector, input, kMaxExtent, 1, "max extent saturates");

  uint32_t state = 0x27B5C4E1u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[40];
    for (int64_t i = 0; i < kMaxExtent; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "rms64 trial %d", trial);
    failed |= check(_mlir_ciface_rms64_q15_vector, input, kExtent, 1, label);
    snprintf(label, sizeof label, "rms64 floor trial %d", trial);
    failed |= check(_mlir_ciface_rms64_floor_q15_vector, input, kExtent, 0, label);
    snprintf(label, sizeof label, "rms4096 trial %d", trial);
    failed |= check(_mlir_ciface_rms4096_q15_vector, input, kMaxExtent, 1, label);
  }
  return failed;
}
