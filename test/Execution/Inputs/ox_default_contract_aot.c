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

extern void _mlir_ciface_q15_fir_filter_default_contract(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_q15_fir_filter_explicit_contract(MemRefI16 *, MemRefI16 *, MemRefI16 *);

enum { kInput = 64, kTaps = 8, kOutput = 57, kTrialCount = 6 };

/* The default picks an inferred i35 wrapping accumulator and the explicit
 * spelling an i40 saturating one. Eight Q15 products bound the window sum by
 * 8 * 2^30 = 2^33, so neither mode is reachable and both programs are the
 * exact sum followed by one ties-positive saturating export (the language's
 * export default, which the explicit twin spells out). This reference
 * computes that exact sum in i64 and is what both must equal. */
static int16_t referenceOutput(const int16_t *input, const int16_t *taps, int64_t index) {
  int64_t sum = 0;
  for (int64_t k = 0; k < kTaps; ++k)
    sum += (int64_t)input[index + k] * (int64_t)taps[k];
  int64_t quotient = sum >> 15;
  int64_t remainder = sum - (quotient << 15);
  int64_t half = (int64_t)1 << 14;
  if (remainder >= half)
    ++quotient;
  if (quotient > 32767)
    quotient = 32767;
  if (quotient < -32768)
    quotient = -32768;
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

int main(void) {
  int failed = 0;
  uint32_t state = 0x2F6B37A1u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t input[kInput];
    int16_t taps[kTaps];
    for (int64_t i = 0; i < kInput; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    for (int64_t k = 0; k < kTaps; ++k) {
      state = nextState(state);
      taps[k] = toSigned16(state);
    }
    /* The rail corpus is where a narrower accumulator or a different update
     * mode would first show: every product reaches -2^30. */
    if (trial == 0) {
      for (int64_t i = 0; i < kInput; ++i)
        input[i] = INT16_MIN;
      for (int64_t k = 0; k < kTaps; ++k)
        taps[k] = INT16_MIN;
    }
    if (trial == 1) {
      for (int64_t i = 0; i < kInput; ++i)
        input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
      for (int64_t k = 0; k < kTaps; ++k)
        taps[k] = INT16_MAX;
    }

    MemRefI16 inputRef = {input, input, 0, {kInput}, {1}};
    MemRefI16 tapsRef = {taps, taps, 0, {kTaps}, {1}};
    MemRefI16 byDefault;
    MemRefI16 explicitly;
    _mlir_ciface_q15_fir_filter_default_contract(&byDefault, &inputRef, &tapsRef);
    _mlir_ciface_q15_fir_filter_explicit_contract(&explicitly, &inputRef, &tapsRef);
    for (int64_t i = 0; i < kOutput; ++i) {
      int16_t expected = referenceOutput(input, taps, i);
      int16_t defaulted = byDefault.aligned[byDefault.offset + i];
      int16_t spelled = explicitly.aligned[explicitly.offset + i];
      if (defaulted != expected || spelled != expected) {
        fprintf(stderr, "trial %d output %lld: default %d, explicit %d, expected %d\n", trial,
                (long long)i, defaulted, spelled, expected);
        failed = 1;
      }
    }
    free(byDefault.allocated);
    free(explicitly.allocated);
  }
  return failed;
}
