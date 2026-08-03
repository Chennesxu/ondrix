#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the f32 Goertzel contract. off and fma are exact, so those
 * are bit for bit against a reference that derives the same coefficient and
 * runs the same event graph; fast is checked for membership in the derivable
 * set, which here has two elements. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

extern void _mlir_ciface_f32_goertzel_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_goertzel_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_goertzel_fast(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_goertzel_quarter_turn(MemRefF32Rank1 *, MemRefF32Rank1 *);

enum { kLength = 16, kBin = 3, kQuarterTurnBin = 4, kTrialCount = 32 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* Mirrors the lowering's quarter-turn evaluation. This does not gate the
 * snap: the unsnapped 6.1e-17 perturbs no exported bit at these extents, so
 * only the conversion test pins it. */
static float doubledCoefficient(int64_t bin) {
  static const double kTwoPi = 6.28318530717958647692528676655900577;
  static const double kQuarterTurns[4] = {1.0, 0.0, -1.0, 0.0};
  double cosine = 4 * bin % kLength == 0 ? kQuarterTurns[(4 * bin / kLength) % 4]
                                         : cos(kTwoPi * (double)bin / (double)kLength);
  return 2.0f * (float)cosine;
}

/* Independent of the lowering apart from the declared event order. */
static float referenceGoertzel(const float *input, int64_t bin, int fused) {
  const float c2 = doubledCoefficient(bin);
  float s1 = 0.0f;
  float s2 = 0.0f;
  for (int64_t n = 0; n < kLength; ++n) {
    const float combined = fused ? fmaf(c2, s1, input[n]) : input[n] + c2 * s1;
    const float next = combined - s2;
    s2 = s1;
    s1 = next;
  }
  const float m = c2 * s1;
  return (s1 * s1 + s2 * s2) - m * s2;
}

static int compare(const char *label, const char *mode, float got, float expected) {
  if (floatBits(got) == floatBits(expected))
    return 0;
  fprintf(stderr, "%s %s: got %a, expected %a\n", label, mode, (double)got, (double)expected);
  return 1;
}

static int check(const float *input, const char *label) {
  float copy[kLength];
  memcpy(copy, input, sizeof(copy));
  MemRefF32Rank1 inputRef = {copy, copy, 0, {kLength}, {1}};
  MemRefF32Rank1 off, fma, fast, quarter;
  _mlir_ciface_f32_goertzel_off(&off, &inputRef);
  _mlir_ciface_f32_goertzel_fma(&fma, &inputRef);
  _mlir_ciface_f32_goertzel_fast(&fast, &inputRef);
  _mlir_ciface_f32_goertzel_quarter_turn(&quarter, &inputRef);

  const float offValue = off.aligned[off.offset];
  const float fmaValue = fma.aligned[fma.offset];
  const float fastValue = fast.aligned[fast.offset];
  int failed = 0;
  failed |= compare(label, "goertzel off", offValue, referenceGoertzel(input, kBin, 0));
  failed |= compare(label, "goertzel fma", fmaValue, referenceGoertzel(input, kBin, 1));
  /* The recursion has no reduction to reassociate, so the one multiply-add
   * site is the whole of what fast may vary: it is free to be fused or not
   * and nothing else can move. That derivable set has two elements, so this
   * checks membership in it rather than a numeric bound. */
  if (floatBits(fastValue) != floatBits(offValue) && floatBits(fastValue) != floatBits(fmaValue)) {
    fprintf(stderr, "%s goertzel fast: got %a, derivable are %a and %a\n", label, (double)fastValue,
            (double)offValue, (double)fmaValue);
    failed = 1;
  }
  failed |= compare(label, "goertzel quarter turn", quarter.aligned[quarter.offset],
                    referenceGoertzel(input, kQuarterTurnBin, 0));
  free(off.allocated);
  free(fma.allocated);
  free(fast.allocated);
  free(quarter.allocated);
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
  const int16_t raw = (int16_t)(nextRandom(state) >> 16);
  return (float)raw / 8192.0f;
}

/* A directed split witness: fusing an off term or splitting an fma term
 * changes the exported energy. */
static int checkContractSplit(void) {
  /* Full-mantissa samples against a full-mantissa coefficient make every
   * recursion product inexact. */
  float input[kLength] = {0};
  input[0] = 0x1.3c6ef3p+0f;
  input[1] = -0x1.1e2d5bp+0f;
  input[2] = 0x1.7a4c9dp+0f;
  return check(input, "contract split");
}

int main(void) {
  int failed = checkContractSplit();

  float input[kLength];
  uint32_t state = UINT32_C(0x6C078965);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    snprintf(label, sizeof label, "trial %d", trial);
    for (int i = 0; i < kLength; ++i)
      input[i] = randomValue(&state);
    failed |= check(input, label);
  }
  return failed;
}
