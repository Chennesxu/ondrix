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

/* CIC_SOURCE_SYMBOL selects the .ox chain leg, where one kernel carries the
 * s2/r4/m1 wrap profile and the same corpus and reference apply. */
#ifdef CIC_SOURCE_SYMBOL
extern void CIC_SOURCE_SYMBOL(MemRefI16 *, MemRefI16 *);
#else
extern void _mlir_ciface_cic_s1_r2_m1_wrap(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_cic_s2_r4_m1_wrap(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_cic_s2_r4_m1_saturate(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_cic_s3_r8_m2_wrap(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_cic_s4_r16_m1_wrap(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_cic_s8_r64_m1_wrap(MemRefI16 *, MemRefI16 *);
#endif

enum { kMaxInput = 128, kMaxOutput = 64, kStageLimit = 8, kTrialCount = 8 };

/* Independent reference: the contract's pseudocode with the two overflow
 * modes written out separately, so neither the carrier width nor the mode
 * is inherited from the compiler's own arithmetic. */
/* The exact sum or difference arrives in an __int128 carrier: at the W=64
 * rail the i64 addition itself could overflow before the mode applies. */
static int64_t wrapTo(__int128 value, int width) {
  uint64_t mask = (width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << width) - 1);
  uint64_t bits = (uint64_t)value & mask;
  uint64_t sign = (uint64_t)1 << (width - 1);
  return (bits & sign) ? (int64_t)(bits | ~mask) : (int64_t)bits;
}

static int64_t saturateTo(__int128 value, int width) {
  int64_t high = (int64_t)(((uint64_t)1 << (width - 1)) - 1);
  int64_t low = -high - 1;
  return value > high ? high : (value < low ? low : (int64_t)value);
}

static int64_t combine(__int128 value, int width, int wrapping) {
  return wrapping ? wrapTo(value, width) : saturateTo(value, width);
}

/* The hand-written profiles spell nearest_even; the .ox chain leg carries
 * the language's export default and selects ties-positive here. */
#ifndef CIC_EXPORT_TIES_POSITIVE
#define CIC_EXPORT_TIES_POSITIVE 0
#endif

static int16_t exportSample(int64_t value, int shift) {
  int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  int64_t half = divisor >> 1;
  if (CIC_EXPORT_TIES_POSITIVE ? remainder >= half
                               : (remainder > half || (remainder == half && (quotient & 1))))
    ++quotient;
  if (quotient > 32767)
    quotient = 32767;
  if (quotient < -32768)
    quotient = -32768;
  return (int16_t)quotient;
}

static int log2Exact(int value) {
  int bits = 0;
  while (value > 1) {
    value >>= 1;
    ++bits;
  }
  return bits;
}

static void reference(const int16_t *input, int64_t outputs, int stages, int rate, int delay,
                      int wrapping, int16_t *result) {
  int growth = stages * log2Exact(rate * delay);
  int width = 16 + growth;
  int64_t integrator[kStageLimit] = {0};
  int64_t line[kStageLimit][2] = {{0}};
  for (int64_t block = 0; block < outputs; ++block) {
    int64_t carried = 0;
    for (int64_t phase = 0; phase < rate; ++phase) {
      carried = input[block * rate + phase];
      for (int stage = 0; stage < stages; ++stage) {
        integrator[stage] = combine((__int128)integrator[stage] + carried, width, wrapping);
        carried = integrator[stage];
      }
    }
    for (int stage = 0; stage < stages; ++stage) {
      int64_t differenced = combine((__int128)carried - line[stage][delay - 1], width, wrapping);
      for (int tap = delay - 1; tap > 0; --tap)
        line[stage][tap] = line[stage][tap - 1];
      line[stage][0] = carried;
      carried = differenced;
    }
    result[block] = exportSample(carried, growth);
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

static int check(void (*kernel)(MemRefI16 *, MemRefI16 *), const int16_t *input, int64_t outputs,
                 int stages, int rate, int delay, int wrapping, const char *label) {
  int64_t inputLength = outputs * rate;
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {inputLength}, {1}};
  MemRefI16 output;
  kernel(&output, &inputRef);

  int16_t expected[kMaxOutput];
  reference(input, outputs, stages, rate, delay, wrapping, expected);
  int failed = output.sizes[0] != outputs;
  for (int64_t i = 0; i < outputs && i < output.sizes[0]; ++i) {
    int16_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s output %lld: got %d, expected %d\n", label, (long long)i, actual,
              expected[i]);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

/* Steady-state identity: the exact DC gain is 2^growth and the export
 * divides by exactly that, so a constant input must reappear unchanged once
 * the combs have filled after stages*delay outputs. Under wrap this holds
 * even though every integrator has long since left the carrier range; it is
 * the property that makes the mode part of the contract. */
static int checkConstantRecovery(void (*kernel)(MemRefI16 *, MemRefI16 *), int16_t level,
                                 int64_t outputs, int stages, int rate, int delay,
                                 const char *label) {
  int16_t input[kMaxInput];
  for (int64_t i = 0; i < outputs * rate; ++i)
    input[i] = level;
  MemRefI16 inputRef = {input, input, 0, {outputs * rate}, {1}};
  MemRefI16 output;
  kernel(&output, &inputRef);
  int failed = 0;
  for (int64_t i = (int64_t)stages * delay; i < outputs; ++i) {
    int16_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != level) {
      fprintf(stderr, "%s steady output %lld: got %d, expected %d\n", label, (long long)i, actual,
              level);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

#ifndef CIC_SOURCE_SYMBOL
/* The mode gate. A sustained full-scale input drives the integrators past
 * the carrier; the two kernels are otherwise identical programs. If they
 * ever agree here the divergence corpus has stopped witnessing anything. */
static int checkModeDivergence(void) {
  int16_t input[kMaxInput];
  for (int64_t i = 0; i < 32; ++i)
    input[i] = 32767;
  MemRefI16 inputRef = {input, input, 0, {32}, {1}};
  MemRefI16 wrapped;
  MemRefI16 saturated;
  _mlir_ciface_cic_s2_r4_m1_wrap(&wrapped, &inputRef);
  _mlir_ciface_cic_s2_r4_m1_saturate(&saturated, &inputRef);
  int differs = 0;
  for (int64_t i = 0; i < 8; ++i)
    if (wrapped.aligned[wrapped.offset + i] != saturated.aligned[saturated.offset + i])
      differs = 1;
  if (!differs)
    fprintf(stderr, "wrap and saturate agreed on the full-scale corpus\n");
  free(wrapped.allocated);
  free(saturated.allocated);
  return !differs;
}
#endif

int main(void) {
  int failed = 0;
  uint32_t state = 0x51C9D3B7u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t input[kMaxInput];
    char label[48];
    for (int64_t i = 0; i < kMaxInput; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    if (trial == 0)
      for (int64_t i = 0; i < kMaxInput; ++i)
        input[i] = 32767;
    if (trial == 1)
      for (int64_t i = 0; i < kMaxInput; ++i)
        input[i] = INT16_MIN;
    if (trial == 2)
      for (int64_t i = 0; i < kMaxInput; ++i)
        input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
    if (trial == 3)
      /* A ramp reaches the export tie at several outputs under a shift of
       * one, where nearest-even is the only rule that keeps the mean. */
      for (int64_t i = 0; i < kMaxInput; ++i)
        input[i] = (int16_t)(i - 64);

#ifdef CIC_SOURCE_SYMBOL
    snprintf(label, sizeof label, "source s2r4m1 trial %d", trial);
    failed |= check(CIC_SOURCE_SYMBOL, input, 8, 2, 4, 1, 1, label);
    continue;
#else
    snprintf(label, sizeof label, "s1r2m1 trial %d", trial);
    failed |= check(_mlir_ciface_cic_s1_r2_m1_wrap, input, 16, 1, 2, 1, 1, label);
    snprintf(label, sizeof label, "s2r4m1 trial %d", trial);
    failed |= check(_mlir_ciface_cic_s2_r4_m1_wrap, input, 8, 2, 4, 1, 1, label);
    snprintf(label, sizeof label, "s2r4m1-sat trial %d", trial);
    failed |= check(_mlir_ciface_cic_s2_r4_m1_saturate, input, 8, 2, 4, 1, 0, label);
    snprintf(label, sizeof label, "s3r8m2 trial %d", trial);
    failed |= check(_mlir_ciface_cic_s3_r8_m2_wrap, input, 8, 3, 8, 2, 1, label);
    snprintf(label, sizeof label, "s4r16m1 trial %d", trial);
    failed |= check(_mlir_ciface_cic_s4_r16_m1_wrap, input, 4, 4, 16, 1, 1, label);
    /* growth 48: the W = 64 admission ceiling, where the update sits at the
     * i64 rail and the exact combine needs the oracle's wide carrier. */
    snprintf(label, sizeof label, "s8r64m1 trial %d", trial);
    failed |= check(_mlir_ciface_cic_s8_r64_m1_wrap, input, 2, 8, 64, 1, 1, label);
#endif
  }

#ifdef CIC_SOURCE_SYMBOL
  failed |= checkConstantRecovery(CIC_SOURCE_SYMBOL, 32767, 8, 2, 4, 1, "source");
  return failed;
#else
  failed |= checkConstantRecovery(_mlir_ciface_cic_s1_r2_m1_wrap, 32767, 16, 1, 2, 1, "s1r2m1");
  failed |= checkConstantRecovery(_mlir_ciface_cic_s2_r4_m1_wrap, 32767, 8, 2, 4, 1, "s2r4m1");
  failed |= checkConstantRecovery(_mlir_ciface_cic_s2_r4_m1_wrap, -32768, 8, 2, 4, 1, "s2r4m1neg");
  failed |= checkConstantRecovery(_mlir_ciface_cic_s3_r8_m2_wrap, 12345, 8, 3, 8, 2, "s3r8m2");
  failed |= checkModeDivergence();
  return failed;
#endif
}
