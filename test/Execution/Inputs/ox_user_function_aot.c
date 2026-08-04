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

extern void _mlir_ciface_q15_call_forward(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_q15_call_inlined(MemRefI16 *, MemRefI16 *, MemRefI16 *);

enum { kInput = 64, kTaps = 8, kOutput = 57, kTrialCount = 6 };

/* Calling a named body must be the same program as writing that body at the
 * call site, contract included. The reference is the declared i40 saturating
 * accumulation with one nearest-even saturating export. */
static int16_t referenceOutput(const int16_t *input, const int16_t *taps, int64_t index) {
  int64_t sum = 0;
  for (int64_t k = 0; k < kTaps; ++k)
    sum += (int64_t)input[index + k] * (int64_t)taps[k];
  int64_t quotient = sum >> 15;
  int64_t remainder = sum - (quotient << 15);
  int64_t half = (int64_t)1 << 14;
  if (remainder > half || (remainder == half && (quotient & 1)))
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
  uint32_t state = 0x71BD4C09u;
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
    if (trial == 0) {
      for (int64_t i = 0; i < kInput; ++i)
        input[i] = INT16_MIN;
      for (int64_t k = 0; k < kTaps; ++k)
        taps[k] = INT16_MIN;
    }

    MemRefI16 inputRef = {input, input, 0, {kInput}, {1}};
    MemRefI16 tapsRef = {taps, taps, 0, {kTaps}, {1}};
    MemRefI16 called;
    MemRefI16 inlined;
    _mlir_ciface_q15_call_forward(&called, &inputRef, &tapsRef);
    _mlir_ciface_q15_call_inlined(&inlined, &inputRef, &tapsRef);
    for (int64_t i = 0; i < kOutput; ++i) {
      int16_t expected = referenceOutput(input, taps, i);
      int16_t viaCall = called.aligned[called.offset + i];
      int16_t viaBody = inlined.aligned[inlined.offset + i];
      if (viaCall != expected || viaBody != expected) {
        fprintf(stderr, "trial %d output %lld: call %d, body %d, expected %d\n", trial,
                (long long)i, viaCall, viaBody, expected);
        failed = 1;
      }
    }
    free(called.allocated);
    free(inlined.allocated);
  }
  return failed;
}
