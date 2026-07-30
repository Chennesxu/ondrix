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

extern void _mlir_ciface_q15_dct8(MemRefI16 *, MemRefI16 *);

enum { kExtent = 8, kShift = 19, kDirectedTrialCount = 7, kRandomTrialCount = 16 };

/* Type-II DCT coefficient table q15(cos(pi*(2n+1)k/(2N))), derived
 * independently with 50-digit mpmath from the contract equation. Identical
 * to the N = 8 table that pins dct_q15_vector_aot.mlir; here it decides the
 * frontend-compiled kernel. */
static const int16_t kDct8[8][8] = {{32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767},
                                    {32138, 27246, 18205, 6393, -6393, -18205, -27246, -32138},
                                    {30274, 12540, -12540, -30274, -30274, -12540, 12540, 30274},
                                    {27246, -6393, -32138, -18205, 18205, 32138, 6393, -27246},
                                    {23170, -23170, -23170, 23170, 23170, -23170, -23170, 23170},
                                    {18205, -32138, 6393, 27246, -27246, -6393, 32138, -18205},
                                    {12540, -30274, 30274, -12540, -12540, 30274, -30274, 12540},
                                    {6393, -18205, 27246, -32138, 32138, -27246, 18205, -6393}};

/* Independent floor-division round-half-even shift by 16 + log2(8) = 19.
 * The saturating i40 accumulator of the compiled kernel is provably never
 * clamped (|sum| <= 8 * 32767 * 32768, about 2^33, far below 2^39), so the
 * exact i64 sum below is the contract value the compiled schedule must
 * reproduce. */
static int16_t roundHalfEvenShift(int64_t value) {
  int64_t divisor = (int64_t)1 << kShift;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  int64_t half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient > 32767)
    return (int16_t)32767;
  if (quotient < -32768)
    return (int16_t)-32768;
  return (int16_t)quotient;
}

static void referenceDct(const int16_t *input, int16_t *output) {
  for (unsigned k = 0; k < kExtent; ++k) {
    int64_t sum = 0;
    for (unsigned n = 0; n < kExtent; ++n)
      sum += (int64_t)input[n] * kDct8[k][n];
    output[k] = roundHalfEvenShift(sum);
  }
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

static void fillDirected(int trial, int16_t *input) {
  for (unsigned i = 0; i < kExtent; ++i)
    input[i] = 0;
  switch (trial) {
  case 0:
    break;
  case 1:
    input[0] = INT16_MAX;
    break;
  case 2:
    input[kExtent - 1] = INT16_MIN;
    break;
  case 3:
    for (unsigned i = 0; i < kExtent; ++i)
      input[i] = INT16_MAX;
    break;
  case 4:
    for (unsigned i = 0; i < kExtent; ++i)
      input[i] = INT16_MIN;
    break;
  case 5:
    for (unsigned i = 0; i < kExtent; ++i)
      input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
    break;
  default:
    for (unsigned i = 0; i < kExtent; ++i)
      input[i] = (i & 1) ? INT16_MAX : INT16_MIN;
    break;
  }
}

static int check(const int16_t *input, const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {kExtent}, {1}};
  MemRefI16 output;
  _mlir_ciface_q15_dct8(&output, &inputRef);

  int16_t expected[kExtent];
  referenceDct(input, expected);

  int failed = output.sizes[0] != kExtent;
  int64_t count = output.sizes[0] < kExtent ? output.sizes[0] : kExtent;
  for (int64_t k = 0; k < count; ++k) {
    int16_t actual = output.aligned[output.offset + k * output.strides[0]];
    if (actual != expected[k]) {
      fprintf(stderr, "%s bin %lld: got %d, expected %d\n", label, (long long)k, actual,
              expected[k]);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

int main(void) {
  int failed = 0;
  int16_t input[kExtent];
  char label[40];

  for (int trial = 0; trial < kDirectedTrialCount; ++trial) {
    fillDirected(trial, input);
    snprintf(label, sizeof label, "directed %d", trial);
    failed |= check(input, label);
  }

  uint32_t state = 0x7C31E94Bu;
  for (int trial = 0; trial < kRandomTrialCount; ++trial) {
    for (unsigned i = 0; i < kExtent; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "random %d", trial);
    failed |= check(input, label);
  }
  return failed;
}
