#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the f32 conv1d contracts. off and fma are exact, so every
 * comparison is bit for bit against a reference that walks the declared
 * traversal itself, in the declared order, for both kernel index maps. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32;

extern void _mlir_ciface_f32_conv1d_conv_off(MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_conv1d_conv_fma(MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_conv1d_conv_fast(MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_conv1d_corr_off(MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_conv1d_corr_fma(MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_conv1d_corr_fast(MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_conv1d_corr_ordered(MemRefF32 *, MemRefF32 *, MemRefF32 *);

enum {
  kLength = 12,
  kTaps = 4,
  kOutputs = 9,
  kTrialCount = 32,
  /* The batched correlation profile: twenty taps at width eight is a seed
   * block, one vector-loop iteration, and a four-element ordered tail. */
  kWideLength = 40,
  kWideTaps = 20,
  kWideOutputs = 21
};

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* Correlation visits kernel[k], convolution kernel[K-1-k]; everything else is
 * shared, so a reversed traversal is the one difference this must catch. */
static float referenceTaps(const float *input, const float *kernel, int64_t taps, int64_t output,
                           int reversed, int fused) {
  float accumulator = 0.0f;
  for (int64_t k = 0; k < taps; ++k) {
    const float tap = reversed ? kernel[taps - 1 - k] : kernel[k];
    const float sample = input[output + k];
    accumulator = fused ? fmaf(sample, tap, accumulator) : accumulator + sample * tap;
  }
  return accumulator;
}

static float reference(const float *input, const float *kernel, int64_t output, int reversed,
                       int fused) {
  return referenceTaps(input, kernel, kTaps, output, reversed, fused);
}

static int compare(const char *label, const char *mode, int64_t index, float got, float expected) {
  if (floatBits(got) == floatBits(expected))
    return 0;
  fprintf(stderr, "%s %s [%lld]: got %a, expected %a\n", label, mode, (long long)index, (double)got,
          (double)expected);
  return 1;
}

static int check(const float *input, const float *kernel, const char *label) {
  float inputCopy[kLength];
  float kernelCopy[kTaps];
  memcpy(inputCopy, input, sizeof(inputCopy));
  memcpy(kernelCopy, kernel, sizeof(kernelCopy));

  MemRefF32 inputRef = {inputCopy, inputCopy, 0, {kLength}, {1}};
  MemRefF32 kernelRef = {kernelCopy, kernelCopy, 0, {kTaps}, {1}};
  MemRefF32 convOff, convFma, corrOff, corrFma;
  _mlir_ciface_f32_conv1d_conv_off(&convOff, &inputRef, &kernelRef);
  _mlir_ciface_f32_conv1d_conv_fma(&convFma, &inputRef, &kernelRef);
  _mlir_ciface_f32_conv1d_corr_off(&corrOff, &inputRef, &kernelRef);
  _mlir_ciface_f32_conv1d_corr_fma(&corrFma, &inputRef, &kernelRef);

  int failed = 0;
  for (int64_t n = 0; n < kOutputs; ++n) {
    const float convExpectedOff = reference(input, kernel, n, 1, 0);
    const float convExpectedFma = reference(input, kernel, n, 1, 1);
    failed |= compare(label, "conv off", n, convOff.aligned[convOff.offset + n], convExpectedOff);
    failed |= compare(label, "conv fma", n, convFma.aligned[convFma.offset + n], convExpectedFma);
    failed |= compare(label, "corr off", n, corrOff.aligned[corrOff.offset + n],
                      reference(input, kernel, n, 0, 0));
    failed |= compare(label, "corr fma", n, corrFma.aligned[corrFma.offset + n],
                      reference(input, kernel, n, 0, 1));
  }
  free(convOff.allocated);
  free(convFma.allocated);
  free(corrOff.allocated);
  free(corrFma.allocated);
  return failed;
}

static uint32_t nextRandom(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

static float randomValue(uint32_t *state) {
  /* Full-mantissa magnitudes near one, so products are inexact and the off and
   * fma legs actually separate. */
  const uint32_t bits = nextRandom(state);
  const float unit = (float)(bits >> 8) * 0x1p-24f;
  return (bits & 1u ? -1.0f : 1.0f) * (0.5f + unit);
}

/* The off and fma legs only separate where a product is inexact and a later
 * cancellation keeps the residual. Term 0 contributes -fl(a*a) exactly and
 * term 1 the exact product a*a: the rounded chain cancels to +0.0 while the
 * fused chain keeps a*a - fl(a*a) = 2^-46. The two modes traverse the kernel in
 * opposite order, so each needs its own corpus, and the separation is asserted
 * rather than assumed - a corpus on which the two legs agree is not evidence
 * for the split, however deliberate its constants look. */
static int checkContractSplit(void) {
  const float a = 0x1.000002p+0f;
  const float rounded = a * a;
  float input[kLength];
  for (int64_t i = 0; i < kLength; ++i)
    input[i] = 0.0f;
  input[0] = -rounded;
  input[1] = a;

  float correlationKernel[kTaps] = {1.0f, a, 0.0f, 0.0f};
  float convolutionKernel[kTaps] = {0.0f, 0.0f, a, 1.0f};

  int failed = check(input, correlationKernel, "contract split corr");
  failed |= check(input, convolutionKernel, "contract split conv");

  MemRefF32 inputRef = {input, input, 0, {kLength}, {1}};
  MemRefF32 corrKernelRef = {correlationKernel, correlationKernel, 0, {kTaps}, {1}};
  MemRefF32 convKernelRef = {convolutionKernel, convolutionKernel, 0, {kTaps}, {1}};
  MemRefF32 corrOff, corrFma, convOff, convFma;
  _mlir_ciface_f32_conv1d_corr_off(&corrOff, &inputRef, &corrKernelRef);
  _mlir_ciface_f32_conv1d_corr_fma(&corrFma, &inputRef, &corrKernelRef);
  _mlir_ciface_f32_conv1d_conv_off(&convOff, &inputRef, &convKernelRef);
  _mlir_ciface_f32_conv1d_conv_fma(&convFma, &inputRef, &convKernelRef);
  if (floatBits(corrOff.aligned[corrOff.offset]) == floatBits(corrFma.aligned[corrFma.offset])) {
    fprintf(stderr, "correlation split corpus is vacuous: off and fma agree\n");
    failed = 1;
  }
  if (floatBits(convOff.aligned[convOff.offset]) == floatBits(convFma.aligned[convFma.offset])) {
    fprintf(stderr, "convolution split corpus is vacuous: off and fma agree\n");
    failed = 1;
  }
  free(corrOff.allocated);
  free(corrFma.allocated);
  free(convOff.allocated);
  free(convFma.allocated);
  return failed;
}

/* Both modes at one shape, so the only variable is the kernel index map.
 *
 * Correlation reaches the batched schedule and is a relaxed result, so it is
 * checked for term conservation on an integer sub-domain: every product is a
 * small integer and every partial sum stays under 2^24, so all derivable
 * regroupings are rounding-free and agree on one exact integer, while a
 * dropped, duplicated or misindexed term still changes it. Convolution's
 * reversed subview is refused by the batching rewrite, so it takes the scalar
 * fused route, which is one declared member and can be bit-pinned. */
static int checkWideModes(void) {
  float input[kWideLength];
  float kernel[kWideTaps];
  for (int64_t i = 0; i < kWideLength; ++i)
    input[i] = (float)((i % 7) - 3);
  for (int64_t i = 0; i < kWideTaps; ++i)
    kernel[i] = (float)((i % 5) - 2);

  MemRefF32 inputRef = {input, input, 0, {kWideLength}, {1}};
  MemRefF32 kernelRef = {kernel, kernel, 0, {kWideTaps}, {1}};
  MemRefF32 batched, ordered, reversed;
  _mlir_ciface_f32_conv1d_corr_fast(&batched, &inputRef, &kernelRef);
  _mlir_ciface_f32_conv1d_corr_ordered(&ordered, &inputRef, &kernelRef);
  _mlir_ciface_f32_conv1d_conv_fast(&reversed, &inputRef, &kernelRef);

  int failed = 0;
  for (int64_t n = 0; n < kWideOutputs; ++n) {
    long exact = 0;
    for (int64_t k = 0; k < kWideTaps; ++k)
      exact += (long)((n + k) % 7 - 3) * (long)(k % 5 - 2);
    const float expected = (float)exact;
    failed |=
        compare("integer lattice", "corr fast", n, batched.aligned[batched.offset + n], expected);
    failed |= compare("integer lattice", "corr ordered", n, ordered.aligned[ordered.offset + n],
                      expected);
    failed |= compare("integer lattice", "conv fast", n, reversed.aligned[reversed.offset + n],
                      referenceTaps(input, kernel, kWideTaps, n, 1, 1));
  }
  free(batched.allocated);
  free(ordered.allocated);
  free(reversed.allocated);
  return failed;
}

int main(void) {
  float input[kLength];
  float kernel[kTaps];

  /* An asymmetric kernel, so convolution and correlation must disagree. */
  for (int64_t i = 0; i < kLength; ++i)
    input[i] = (float)(i + 1);
  kernel[0] = 1.0f;
  kernel[1] = 2.0f;
  kernel[2] = 4.0f;
  kernel[3] = 8.0f;
  int failed = check(input, kernel, "asymmetric kernel");

  MemRefF32 inputRef = {input, input, 0, {kLength}, {1}};
  MemRefF32 kernelRef = {kernel, kernel, 0, {kTaps}, {1}};
  MemRefF32 conv, corr;
  _mlir_ciface_f32_conv1d_conv_off(&conv, &inputRef, &kernelRef);
  _mlir_ciface_f32_conv1d_corr_off(&corr, &inputRef, &kernelRef);
  if (floatBits(conv.aligned[conv.offset]) == floatBits(corr.aligned[corr.offset])) {
    fprintf(stderr, "the two modes agree on an asymmetric kernel, so neither is pinned\n");
    failed = 1;
  }
  free(conv.allocated);
  free(corr.allocated);

  failed |= checkContractSplit();

  failed |= checkWideModes();

  uint32_t state = UINT32_C(0x2545F491);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    snprintf(label, sizeof label, "trial %d", trial);
    for (int64_t i = 0; i < kLength; ++i)
      input[i] = randomValue(&state);
    for (int64_t i = 0; i < kTaps; ++i)
      kernel[i] = randomValue(&state);
    failed |= check(input, kernel, label);
  }
  return failed;
}
