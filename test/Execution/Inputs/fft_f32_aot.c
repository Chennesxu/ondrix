#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the interleaved f32 transform profile. The profile has no
 * requantization boundary, so every comparison here is bit for bit against a
 * reference that walks the declared event graph itself. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32;

extern void _mlir_ciface_cfft8_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_cfft8_fma(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_icfft8_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_cfft64_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_cfft64_fma(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_rfft16_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_irfft16_off(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_cfft_round_trip(MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_cfft_round_trip_loops(MemRefF32 *, MemRefF32 *);

enum { kMaxExtent = 64 };

static const double kTwoPi = 6.28318530717958647692528676655900577;

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* The declared recursion: even/odd decimation in time, with no product at
 * index 0 and index n/4 because those twiddles are exactly one and exactly
 * -+j. Fusion is spent only on the two multiply-adds of the complex
 * product. */
static void referenceCfft(const float *inRe, const float *inIm, float *outRe, float *outIm, int n,
                          int forward, int fused) {
  if (n == 1) {
    outRe[0] = inRe[0];
    outIm[0] = inIm[0];
    return;
  }
  const int half = n / 2;
  float evenInRe[kMaxExtent], evenInIm[kMaxExtent];
  float oddInRe[kMaxExtent], oddInIm[kMaxExtent];
  float evenRe[kMaxExtent], evenIm[kMaxExtent];
  float oddRe[kMaxExtent], oddIm[kMaxExtent];
  for (int i = 0; i < half; ++i) {
    evenInRe[i] = inRe[2 * i];
    evenInIm[i] = inIm[2 * i];
    oddInRe[i] = inRe[2 * i + 1];
    oddInIm[i] = inIm[2 * i + 1];
  }
  referenceCfft(evenInRe, evenInIm, evenRe, evenIm, half, forward, fused);
  referenceCfft(oddInRe, oddInIm, oddRe, oddIm, half, forward, fused);
  for (int k = 0; k < half; ++k) {
    float termRe, termIm;
    if (k == 0) {
      termRe = oddRe[k];
      termIm = oddIm[k];
    } else if (k == n / 4) {
      termRe = forward ? oddIm[k] : -oddIm[k];
      termIm = forward ? -oddRe[k] : oddRe[k];
    } else {
      const double angle = kTwoPi * (double)k / (double)n;
      const float wRe = (float)cos(angle);
      const float wIm = (float)(forward ? -sin(angle) : sin(angle));
      const float crossed = -(wIm * oddIm[k]);
      const float straight = wIm * oddRe[k];
      termRe = fused ? fmaf(wRe, oddRe[k], crossed) : crossed + wRe * oddRe[k];
      termIm = fused ? fmaf(wRe, oddIm[k], straight) : straight + wRe * oddIm[k];
    }
    outRe[k] = evenRe[k] + termRe;
    outIm[k] = evenIm[k] + termIm;
    outRe[k + half] = evenRe[k] - termRe;
    outIm[k + half] = evenIm[k] - termIm;
  }
}

static int report(const char *label, int index, float expected, float actual) {
  if (floatBits(expected) == floatBits(actual))
    return 0;
  fprintf(stderr, "%s[%d]: expected 0x%08x, got 0x%08x\n", label, index, floatBits(expected),
          floatBits(actual));
  return 1;
}

/* The kernels return a tensor, so the callee writes the result descriptor and
 * owns the buffer; reading a caller-supplied array would read nothing. */
static float resultAt(const MemRefF32 *result, int index) {
  return result->aligned[result->offset + (int64_t)index * result->strides[0]];
}

static int checkCfft(void (*kernel)(MemRefF32 *, MemRefF32 *), const char *label, int n,
                     int forward, int fused, const float *interleaved) {
  float input[2 * kMaxExtent];
  float inRe[kMaxExtent], inIm[kMaxExtent], outRe[kMaxExtent], outIm[kMaxExtent];
  memcpy(input, interleaved, sizeof(float) * 2 * (size_t)n);
  for (int i = 0; i < n; ++i) {
    inRe[i] = input[2 * i];
    inIm[i] = input[2 * i + 1];
  }
  MemRefF32 inputRef = {input, input, 0, {2 * n}, {1}};
  MemRefF32 result;
  kernel(&result, &inputRef);

  referenceCfft(inRe, inIm, outRe, outIm, n, forward, fused);
  const float scale = forward ? 1.0f : 1.0f / (float)n;
  int failures = 0;
  for (int i = 0; i < n; ++i) {
    failures += report(label, 2 * i, outRe[i] * scale, resultAt(&result, 2 * i));
    failures += report(label, 2 * i + 1, outIm[i] * scale, resultAt(&result, 2 * i + 1));
  }
  free(result.allocated);
  return failures;
}

static int checkRfft(const float *signal, int n) {
  float input[kMaxExtent];
  float inRe[kMaxExtent], inIm[kMaxExtent], outRe[kMaxExtent], outIm[kMaxExtent];
  memcpy(input, signal, sizeof(float) * (size_t)n);
  for (int i = 0; i < n; ++i) {
    inRe[i] = input[i];
    inIm[i] = 0.0f;
  }
  MemRefF32 inputRef = {input, input, 0, {n}, {1}};
  MemRefF32 result;
  _mlir_ciface_rfft16_off(&result, &inputRef);

  referenceCfft(inRe, inIm, outRe, outIm, n, /*forward=*/1, /*fused=*/0);
  outIm[0] = 0.0f;
  outIm[n / 2] = 0.0f;
  int failures = 0;
  for (int i = 0; i <= n / 2; ++i) {
    failures += report("rfft", 2 * i, outRe[i], resultAt(&result, 2 * i));
    failures += report("rfft", 2 * i + 1, outIm[i], resultAt(&result, 2 * i + 1));
  }
  free(result.allocated);
  return failures;
}

static int checkIrfft(const float *bins, int n) {
  float input[kMaxExtent + 2];
  float inRe[kMaxExtent], inIm[kMaxExtent], outRe[kMaxExtent], outIm[kMaxExtent];
  memcpy(input, bins, sizeof(float) * (size_t)(n + 2));
  inRe[0] = input[0];
  inIm[0] = 0.0f;
  inRe[n / 2] = input[n];
  inIm[n / 2] = 0.0f;
  for (int k = 1; k < n / 2; ++k) {
    inRe[k] = input[2 * k];
    inIm[k] = input[2 * k + 1];
    inRe[n - k] = input[2 * k];
    inIm[n - k] = -input[2 * k + 1];
  }
  MemRefF32 inputRef = {input, input, 0, {n + 2}, {1}};
  MemRefF32 result;
  _mlir_ciface_irfft16_off(&result, &inputRef);

  referenceCfft(inRe, inIm, outRe, outIm, n, /*forward=*/0, /*fused=*/0);
  int failures = 0;
  for (int i = 0; i < n; ++i)
    failures += report("irfft", i, outRe[i] * (1.0f / (float)n), resultAt(&result, i));
  free(result.allocated);
  return failures;
}

/* The contract axis has to be observable, or the two modes are one profile
 * under two names. */
static int checkContractsDiffer(const float *interleaved) {
  float inputOff[128], inputFma[128];
  memcpy(inputOff, interleaved, sizeof(inputOff));
  memcpy(inputFma, interleaved, sizeof(inputFma));
  MemRefF32 offIn = {inputOff, inputOff, 0, {128}, {1}};
  MemRefF32 fmaIn = {inputFma, inputFma, 0, {128}, {1}};
  MemRefF32 off, fma;
  _mlir_ciface_cfft64_off(&off, &offIn);
  _mlir_ciface_cfft64_fma(&fma, &fmaIn);
  int differs = 0;
  for (int i = 0; i < 128; ++i)
    if (floatBits(resultAt(&off, i)) != floatBits(resultAt(&fma, i)))
      differs = 1;
  free(off.allocated);
  free(fma.allocated);
  if (differs)
    return 0;
  fprintf(stderr, "off and fma agreed on every lane: the contract axis is not observable\n");
  return 1;
}

/* The .ox binding contributes the source-level spelling of the same contract,
 * and its forward-then-inverse composition is gated as one program. */
static int checkSourceRoundTrip(const float *interleaved) {
  enum { kPoints = 16 };
  float input[2 * kPoints];
  float inRe[kPoints], inIm[kPoints];
  float forwardRe[kPoints], forwardIm[kPoints], inverseRe[kPoints], inverseIm[kPoints];
  memcpy(input, interleaved, sizeof(input));
  for (int i = 0; i < kPoints; ++i) {
    inRe[i] = input[2 * i];
    inIm[i] = input[2 * i + 1];
  }
  MemRefF32 inputRef = {input, input, 0, {2 * kPoints}, {1}};
  MemRefF32 result;
  _mlir_ciface_f32_cfft_round_trip(&result, &inputRef);

  referenceCfft(inRe, inIm, forwardRe, forwardIm, kPoints, /*forward=*/1, /*fused=*/0);
  referenceCfft(forwardRe, forwardIm, inverseRe, inverseIm, kPoints, /*forward=*/0, /*fused=*/0);
  const float scale = 1.0f / (float)kPoints;
  int failures = 0;
  for (int i = 0; i < kPoints; ++i) {
    failures += report("ox", 2 * i, inverseRe[i] * scale, resultAt(&result, 2 * i));
    failures += report("ox", 2 * i + 1, inverseIm[i] * scale, resultAt(&result, 2 * i + 1));
  }
  free(result.allocated);
  return failures;
}

/* `--fft-loops` is an explicit schedule choice, so it must change the code
 * shape and NOT the values. The shape half is checked in
 * test/Frontend/fft_loops_flag.mlir; this is the value half, on the same
 * source compiled both ways. */
static int checkSourceShapesAgree(const float *interleaved) {
  enum { kPoints = 16 };
  float unrolledIn[2 * kPoints], loopedIn[2 * kPoints];
  memcpy(unrolledIn, interleaved, sizeof(unrolledIn));
  memcpy(loopedIn, interleaved, sizeof(loopedIn));
  MemRefF32 unrolledRef = {unrolledIn, unrolledIn, 0, {2 * kPoints}, {1}};
  MemRefF32 loopedRef = {loopedIn, loopedIn, 0, {2 * kPoints}, {1}};
  MemRefF32 unrolled, looped;
  _mlir_ciface_f32_cfft_round_trip(&unrolled, &unrolledRef);
  _mlir_ciface_f32_cfft_round_trip_loops(&looped, &loopedRef);
  int failures = 0;
  for (int i = 0; i < 2 * kPoints; ++i)
    failures += report("ox_loops", i, resultAt(&unrolled, i), resultAt(&looped, i));
  free(unrolled.allocated);
  free(looped.allocated);
  return failures;
}

int main(void) {
  float signal[2 * kMaxExtent];
  for (int i = 0; i < 2 * kMaxExtent; ++i)
    signal[i] = (float)((i * 37 % 61) - 30) * 0.3173828125f + (float)(i % 7) * 0.0009765625f;

  int failures = 0;
  failures += checkCfft(_mlir_ciface_cfft8_off, "cfft8_off", 8, 1, 0, signal);
  failures += checkCfft(_mlir_ciface_cfft8_fma, "cfft8_fma", 8, 1, 1, signal);
  failures += checkCfft(_mlir_ciface_icfft8_off, "icfft8_off", 8, 0, 0, signal);
  failures += checkCfft(_mlir_ciface_cfft64_off, "cfft64_off", 64, 1, 0, signal);
  failures += checkCfft(_mlir_ciface_cfft64_fma, "cfft64_fma", 64, 1, 1, signal);
  failures += checkRfft(signal, 16);
  failures += checkIrfft(signal, 16);
  failures += checkSourceRoundTrip(signal);
  failures += checkSourceShapesAgree(signal);
  failures += checkContractsDiffer(signal);

  if (failures != 0) {
    fprintf(stderr, "%d f32 transform mismatches\n", failures);
    return 1;
  }
  printf("f32 transform gate ok\n");
  return 0;
}
