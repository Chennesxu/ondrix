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

extern void _mlir_ciface_moving_average2_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_moving_average3_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_moving_average6_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_moving_average8_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_moving_average64_q15(MemRefI16 *, MemRefI16 *);

enum { kMaxLength = 64, kTrialCount = 12 };

/* Exact contract: window sum in i64, one round-half-even division by K.
 * The mean of Q1.15 values stays in Q1.15, so no saturation. The half test
 * compares the remainder against window - remainder rather than against
 * floor(window / 2): for an odd window the exact half K/2 is not an
 * integer, so no remainder is a tie there, and the floored comparison
 * would wrongly treat remainder == (K-1)/2 as one. */
static int16_t referenceMean(const int16_t *input, int64_t index, int64_t window) {
  int64_t sum = 0;
  for (int64_t i = 0; i < window; ++i)
    sum += input[index + i];
  int64_t quotient = sum / window;
  int64_t remainder = sum % window;
  if (remainder < 0) {
    --quotient;
    remainder += window;
  }
  int64_t complement = window - remainder;
  if (remainder > complement || (remainder == complement && (quotient & 1)))
    ++quotient;
  return (int16_t)quotient;
}

/* The equal-tap Q15 FIR reformulation quantizes 1/K first. For K = 2^m
 * the tap q15(1/K) = 2^(15-m) makes the scaling exact and the programs
 * coincide, but for a general window the reformulation is a different
 * program: with K = 3, q15(1/3) = 10923 and x = {16385, 16385, 16385},
 * the tap form rounds to 16386 while the exact mean is 16385. The
 * directed trial below runs that exact input through the K = 3 kernel, so
 * the gate itself witnesses which of the two programs this contract is. */

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

static int check(void (*kernel)(MemRefI16 *, MemRefI16 *), const int16_t *input,
                 int64_t inputLength, int64_t window, const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {inputLength}, {1}};
  MemRefI16 output;
  kernel(&output, &inputRef);

  int64_t outputLength = inputLength - window + 1;
  int failed = output.sizes[0] != outputLength;
  int64_t count = output.sizes[0] < outputLength ? output.sizes[0] : outputLength;
  for (int64_t i = 0; i < count; ++i) {
    int16_t expected = referenceMean(input, i, window);
    int16_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected) {
      fprintf(stderr, "%s output %lld: got %d, expected %d\n", label, (long long)i, actual,
              expected);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

int main(void) {
  int failed = 0;
  uint32_t state = 0x0A6EA6E5u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t input[kMaxLength];
    char label[40];
    for (int64_t i = 0; i < kMaxLength; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    if (trial == 0)
      for (int64_t i = 0; i < kMaxLength; ++i)
        input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
    if (trial == 1)
      for (int64_t i = 0; i < kMaxLength; ++i)
        input[i] = INT16_MIN; /* mean is exactly -32768: fits i16 */
    if (trial == 2)
      for (int64_t i = 0; i < kMaxLength; ++i)
        input[i] = INT16_MAX;
    if (trial == 3)
      /* K = 2 half-remainder ties with both quotient parities: {2k, 2k+1}
       * windows land exactly on the rounding half and must round to even.
       * The same ramp reaches the K = 6 tie (remainder 3) as well. */
      for (int64_t i = 0; i < kMaxLength; ++i)
        input[i] = (int16_t)(i - 32);
    if (trial == 4)
      /* The FIR-divergence witness pinned in the comment above. */
      for (int64_t i = 0; i < kMaxLength; ++i)
        input[i] = 16385;
    snprintf(label, sizeof label, "window2 trial %d", trial);
    failed |= check(_mlir_ciface_moving_average2_q15, input, 40, 2, label);
    /* The odd window has no reachable tie; the even non-power-of-two one
     * does. Together they cover both halves of the round_div boundary. */
    snprintf(label, sizeof label, "window3 trial %d", trial);
    failed |= check(_mlir_ciface_moving_average3_q15, input, 40, 3, label);
    snprintf(label, sizeof label, "window6 trial %d", trial);
    failed |= check(_mlir_ciface_moving_average6_q15, input, 40, 6, label);
    snprintf(label, sizeof label, "window8 trial %d", trial);
    failed |= check(_mlir_ciface_moving_average8_q15, input, 40, 8, label);
    /* Maximum window with input length == window: exactly one output. */
    snprintf(label, sizeof label, "window64 trial %d", trial);
    failed |= check(_mlir_ciface_moving_average64_q15, input, 64, 64, label);
  }
  return failed;
}
