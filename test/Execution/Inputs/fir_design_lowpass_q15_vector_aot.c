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

extern void _mlir_ciface_fir_design_lowpass_q15(MemRefI16 *, MemRefI16 *);

enum { kInputLength = 64, kTapCount = 9, kOutputLength = 56, kTrialCount = 16 };

/* Golden Q15 taps for the frozen windowed-sinc lowpass profile
 * (N = 9, fc = 1/4), derived independently with 50-digit mpmath from the
 * real-valued contract equation; they are not read from the compiled
 * module, so a compiler-side design or quantization change breaks this
 * gate. sum(|h|) = 35928, so |accumulator| <= 32768 * 35928 < 2^31 and the
 * i40 saturating accumulator can never clamp: exact int64_t accumulation
 * below is bit-equivalent to the contract. */
static const int16_t kGoldenTaps[kTapCount] = {0, -747, 0, 9025, 16384, 9025, 0, -747, 0};

static int16_t referenceOutput(const int16_t *input, int64_t index) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    accumulator += (int64_t)input[index + tap] * (int64_t)kGoldenTaps[tap];

  int64_t quotient = accumulator / 32768;
  int64_t remainder = accumulator % 32768;
  if (remainder < 0) {
    remainder += 32768;
    --quotient;
  }
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
    ++quotient;
  if (quotient < INT16_MIN)
    return INT16_MIN;
  if (quotient > INT16_MAX)
    return INT16_MAX;
  return (int16_t)quotient;
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

static int checkOutput(const int16_t *input, const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {kInputLength}, {1}};
  MemRefI16 output;
  _mlir_ciface_fir_design_lowpass_q15(&output, &inputRef);

  int failed = output.sizes[0] != kOutputLength;
  if (failed)
    fprintf(stderr, "%s: output length %lld\n", label, (long long)output.sizes[0]);
  int64_t count = output.sizes[0] < kOutputLength ? output.sizes[0] : kOutputLength;
  for (int64_t i = 0; i < count; ++i) {
    int16_t expected = referenceOutput(input, i);
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

  int16_t extreme[kInputLength];
  for (int64_t i = 0; i < kInputLength; ++i)
    extreme[i] = (i & 1) ? INT16_MIN : INT16_MAX;
  failed |= checkOutput(extreme, "extreme");

  uint32_t state = 0x0A15D51Bu;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t input[kInputLength];
    char label[32];
    for (int64_t i = 0; i < kInputLength; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "trial %d", trial);
    failed |= checkOutput(input, label);
  }
  return failed;
}
