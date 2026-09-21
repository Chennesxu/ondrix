#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for the interleaved f32 complex FIR. The window is the only
 * thing this adds over the scalar reduction, so the reference reduces each
 * window itself and the comparison is bit for bit. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

extern void _mlir_ciface_cx_fir_f32_off(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_fir_f32_conj_fma(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void ondrix_f32_cx_fir(const float *, const float *, float *);

enum { kSamples = 12, kTaps = 4, kOutputs = kSamples - kTaps + 1, kTrialCount = 64 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static void referenceWindow(const float *x, const float *h, int conjugate, int fused, float *out) {
  float real = 0.0f, imaginary = 0.0f;
  for (int64_t tap = 0; tap < kTaps; ++tap) {
    const float xr = x[2 * tap], xi = x[2 * tap + 1];
    const float hr = h[2 * tap], hi = h[2 * tap + 1];
    float realTerm, imagTerm;
    if (conjugate) {
      realTerm = fused ? fmaf(xi, hi, xr * hr) : xr * hr + xi * hi;
      imagTerm = fused ? fmaf(-xr, hi, xi * hr) : xi * hr - xr * hi;
    } else {
      realTerm = fused ? fmaf(-xi, hi, xr * hr) : xr * hr - xi * hi;
      imagTerm = fused ? fmaf(xi, hr, xr * hi) : xr * hi + xi * hr;
    }
    real = real + realTerm;
    imaginary = imaginary + imagTerm;
  }
  out[0] = real;
  out[1] = imaginary;
}

static void fill(float *values, int64_t count, uint32_t *state) {
  for (int64_t index = 0; index < count; ++index) {
    *state = *state * 1664525u + 1013904223u;
    values[index] = (float)((int32_t)(*state >> 8) - (1 << 23)) * 0x1p-24f;
  }
}

static int checkTrial(uint32_t trial) {
  float signal[2 * kSamples], taps[2 * kTaps];
  float input[2 * kSamples], coeffs[2 * kTaps], got[2 * kOutputs], want[2];
  uint32_t state = trial * 2654435761u + 31u;
  fill(signal, 2 * kSamples, &state);
  fill(taps, 2 * kTaps, &state);

  int failed = 0;
  for (int conjugate = 0; conjugate < 2; ++conjugate) {
    const int fused = conjugate; /* off plain, fma conjugate: the two kernels */
    memcpy(input, signal, sizeof(input));
    memcpy(coeffs, taps, sizeof(coeffs));
    MemRefF32Rank1 inputRef = {input, input, 0, {2 * kSamples}, {1}};
    MemRefF32Rank1 coeffRef = {coeffs, coeffs, 0, {2 * kTaps}, {1}};
    MemRefF32Rank1 outRef = {got, got, 0, {2 * kOutputs}, {1}};
    (conjugate ? _mlir_ciface_cx_fir_f32_conj_fma
               : _mlir_ciface_cx_fir_f32_off)(&inputRef, &coeffRef, &outRef);
    for (int64_t output = 0; output < kOutputs; ++output) {
      referenceWindow(signal + 2 * output, taps, conjugate, fused, want);
      for (int component = 0; component < 2; ++component) {
        if (floatBits(got[2 * output + component]) != floatBits(want[component])) {
          printf("%s trial %u output %lld component %d: got %a, want %a\n",
                 conjugate ? "conjugate fma" : "plain off", trial, (long long)output, component,
                 (double)got[2 * output + component], (double)want[component]);
          failed = 1;
        }
      }
    }
  }
  return failed;
}

/* The source spelling is the conjugate fma kernel, so it must agree with it
 * bit for bit rather than merely be close. */
static int checkSource(void) {
  float signal[2 * kSamples], taps[2 * kTaps];
  float input[2 * kSamples], coeffs[2 * kTaps], staged[2 * kOutputs], source[2 * kOutputs];
  int failed = 0;
  for (uint32_t trial = 1; trial <= 32; ++trial) {
    uint32_t state = trial * 40503u + 5u;
    fill(signal, 2 * kSamples, &state);
    fill(taps, 2 * kTaps, &state);
    memcpy(input, signal, sizeof(input));
    memcpy(coeffs, taps, sizeof(coeffs));
    MemRefF32Rank1 inputRef = {input, input, 0, {2 * kSamples}, {1}};
    MemRefF32Rank1 coeffRef = {coeffs, coeffs, 0, {2 * kTaps}, {1}};
    MemRefF32Rank1 outRef = {staged, staged, 0, {2 * kOutputs}, {1}};
    _mlir_ciface_cx_fir_f32_conj_fma(&inputRef, &coeffRef, &outRef);
    ondrix_f32_cx_fir(signal, taps, source);
    for (int64_t index = 0; index < 2 * kOutputs; ++index) {
      if (floatBits(staged[index]) != floatBits(source[index])) {
        printf("source trial %u element %lld: staged %a, source %a\n", trial, (long long)index,
               (double)staged[index], (double)source[index]);
        failed = 1;
      }
    }
  }
  return failed;
}

int main(void) {
  int failed = 0;
  for (uint32_t trial = 0; trial < kTrialCount; ++trial)
    failed |= checkTrial(trial);
  failed |= checkSource();
  if (failed) {
    printf("interleaved f32 complex FIR FAILED\n");
    return 1;
  }
  printf("interleaved f32 complex FIR OK\n");
  return 0;
}
