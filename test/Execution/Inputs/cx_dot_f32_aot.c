#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for the interleaved f32 complex reduction. The profile has no
 * requantization, so every comparison is bit for bit against a reference that
 * walks the same event graph, and the two contract modes are separated by a
 * named witness rather than assumed to differ. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

extern void _mlir_ciface_cx_dot_f32_off(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_dot_f32_fma(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_corr_f32_off(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_corr_f32_fma(MemRefF32Rank1 *, MemRefF32Rank1 *, MemRefF32Rank1 *);

enum { kValues = 8, kElements = 2 * kValues, kTrialCount = 64 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* The declared pairing, in emission order: each component's two cross
 * products combine into one term before the accumulator update sees it, and
 * the term seeds on its first product. */
static void reference(const float *x, const float *y, int conjugate, int fused, float *out) {
  float real = 0.0f, imaginary = 0.0f;
  for (int64_t index = 0; index < kValues; ++index) {
    const float xr = x[2 * index], xi = x[2 * index + 1];
    const float yr = y[2 * index], yi = y[2 * index + 1];
    float realTerm, imagTerm;
    if (conjugate) {
      realTerm = fused ? fmaf(xi, yi, xr * yr) : xr * yr + xi * yi;
      imagTerm = fused ? fmaf(-xr, yi, xi * yr) : xi * yr - xr * yi;
    } else {
      realTerm = fused ? fmaf(-xi, yi, xr * yr) : xr * yr - xi * yi;
      imagTerm = fused ? fmaf(xi, yr, xr * yi) : xr * yi + xi * yr;
    }
    real = real + realTerm;
    imaginary = imaginary + imagTerm;
  }
  out[0] = real;
  out[1] = imaginary;
}

/* Value 0 is the named witness: its two contracts disagree in both the dot
 * and the correlation. The assertion below proves that rather than assuming
 * it, so a lowering that quietly dropped the contract mode would fail here. */
static void fillCorpus(float *x, float *y, uint32_t trial) {
  const float witnessX[2] = {0x1.44961ep-2f, 0x1.01e3d2p-3f};
  const float witnessY[2] = {0x1.725f9cp-1f, -0x1.1578fp-4f};
  x[0] = witnessX[0];
  x[1] = witnessX[1];
  y[0] = witnessY[0];
  y[1] = witnessY[1];
  uint32_t state = trial * 2654435761u + 101u;
  for (int64_t index = 1; index < kValues; ++index) {
    for (int64_t half = 0; half < 2; ++half) {
      state = state * 1664525u + 1013904223u;
      const float mantissa = (float)((int32_t)(state >> 8) - (1 << 23)) * 0x1p-24f;
      x[2 * index + half] = mantissa;
      state = state * 1664525u + 1013904223u;
      y[2 * index + half] = (float)((int32_t)(state >> 8) - (1 << 23)) * 0x1p-24f;
    }
  }
  if (trial == 1) {
    /* One all-zero operand: the reduction must return a signed zero pair,
     * not a NaN, and the conjugate energy of zero is zero. */
    for (int64_t index = 0; index < kElements; ++index)
      y[index] = 0.0f;
  }
}

static int checkTrial(uint32_t trial) {
  float x[kElements], y[kElements];
  float lhs[kElements], rhs[kElements], got[2], want[2];
  fillCorpus(x, y, trial);
  int failed = 0;
  for (int conjugate = 0; conjugate < 2; ++conjugate) {
    for (int fused = 0; fused < 2; ++fused) {
      memcpy(lhs, x, sizeof(lhs));
      memcpy(rhs, y, sizeof(rhs));
      MemRefF32Rank1 lhsRef = {lhs, lhs, 0, {kElements}, {1}};
      MemRefF32Rank1 rhsRef = {rhs, rhs, 0, {kElements}, {1}};
      MemRefF32Rank1 outRef = {got, got, 0, {2}, {1}};
      if (conjugate)
        (fused ? _mlir_ciface_cx_corr_f32_fma : _mlir_ciface_cx_corr_f32_off)(&lhsRef, &rhsRef,
                                                                              &outRef);
      else
        (fused ? _mlir_ciface_cx_dot_f32_fma : _mlir_ciface_cx_dot_f32_off)(&lhsRef, &rhsRef,
                                                                            &outRef);
      reference(x, y, conjugate, fused, want);
      for (int component = 0; component < 2; ++component) {
        if (floatBits(got[component]) != floatBits(want[component])) {
          printf("%s %s trial %u component %d: got %a, want %a\n",
                 conjugate ? "correlation" : "dot", fused ? "fma" : "off", trial, component,
                 (double)got[component], (double)want[component]);
          failed = 1;
        }
      }
    }
  }
  return failed;
}

/* The conjugate form at lhs == rhs is the energy: a non-negative real part
 * and an imaginary part that is exactly zero term by term. */
static int checkEnergy(void) {
  float x[kElements], y[kElements], buffer[kElements], got[2];
  fillCorpus(x, y, 7);
  memcpy(buffer, x, sizeof(buffer));
  float mirror[kElements];
  memcpy(mirror, x, sizeof(mirror));
  MemRefF32Rank1 lhsRef = {buffer, buffer, 0, {kElements}, {1}};
  MemRefF32Rank1 rhsRef = {mirror, mirror, 0, {kElements}, {1}};
  MemRefF32Rank1 outRef = {got, got, 0, {2}, {1}};
  _mlir_ciface_cx_corr_f32_off(&lhsRef, &rhsRef, &outRef);
  int failed = 0;
  if (!(got[0] > 0.0f)) {
    printf("energy: real part %a is not positive\n", (double)got[0]);
    failed = 1;
  }
  if (got[1] != 0.0f) {
    printf("energy: imaginary part %a is not zero\n", (double)got[1]);
    failed = 1;
  }
  return failed;
}

static int checkWitnessDiscriminates(void) {
  float x[kElements], y[kElements], lhs[kElements], rhs[kElements];
  float off[2], fma[2];
  fillCorpus(x, y, 0);
  int failed = 0;
  for (int conjugate = 0; conjugate < 2; ++conjugate) {
    memcpy(lhs, x, sizeof(lhs));
    memcpy(rhs, y, sizeof(rhs));
    MemRefF32Rank1 lhsRef = {lhs, lhs, 0, {kElements}, {1}};
    MemRefF32Rank1 rhsRef = {rhs, rhs, 0, {kElements}, {1}};
    MemRefF32Rank1 offRef = {off, off, 0, {2}, {1}};
    MemRefF32Rank1 fmaRef = {fma, fma, 0, {2}, {1}};
    if (conjugate) {
      _mlir_ciface_cx_corr_f32_off(&lhsRef, &rhsRef, &offRef);
      memcpy(lhs, x, sizeof(lhs));
      memcpy(rhs, y, sizeof(rhs));
      _mlir_ciface_cx_corr_f32_fma(&lhsRef, &rhsRef, &fmaRef);
    } else {
      _mlir_ciface_cx_dot_f32_off(&lhsRef, &rhsRef, &offRef);
      memcpy(lhs, x, sizeof(lhs));
      memcpy(rhs, y, sizeof(rhs));
      _mlir_ciface_cx_dot_f32_fma(&lhsRef, &rhsRef, &fmaRef);
    }
    if (floatBits(off[0]) == floatBits(fma[0]) && floatBits(off[1]) == floatBits(fma[1])) {
      printf("%s: the corpus does not separate the two contracts\n",
             conjugate ? "correlation" : "dot");
      failed = 1;
    }
  }
  return failed;
}

int main(void) {
  int failed = checkWitnessDiscriminates();
  for (uint32_t trial = 0; trial < kTrialCount; ++trial)
    failed |= checkTrial(trial);
  failed |= checkEnergy();
  if (failed) {
    printf("interleaved f32 complex reduction FAILED\n");
    return 1;
  }
  printf("interleaved f32 complex reduction OK\n");
  return 0;
}
