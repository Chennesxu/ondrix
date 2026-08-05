#include "goertzel_f32_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

extern void _mlir_ciface_f32_goertzel_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_goertzel_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);

enum { kLength = 16, kBin = 3, kTrialCount = 16 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int compare(const char *label, const char *mode, float got, float expected) {
  if (floatBits(got) == floatBits(expected))
    return 0;
  fprintf(stderr, "%s %s: got %a, expected %a\n", label, mode, (double)got, (double)expected);
  return 1;
}

/* Both source contracts are exact, so each object is checked bit for bit
 * against the graph its own call site declared. `split` counts the inputs on
 * which the two graphs disagree, which is what makes carrying both
 * non-vacuous. */
static int check(const float *input, const char *label, int64_t *split) {
  float copy[kLength];
  memcpy(copy, input, sizeof(copy));
  MemRefF32Rank1 inputRef = {copy, copy, 0, {kLength}, {1}};
  MemRefF32Rank1 off, fma;
  _mlir_ciface_f32_goertzel_off(&off, &inputRef);
  _mlir_ciface_f32_goertzel_fma(&fma, &inputRef);

  const float offValue = off.aligned[off.offset];
  const float fmaValue = fma.aligned[fma.offset];
  int failed = compare(label, "goertzel off", offValue, goertzelReference(input, kLength, kBin, 0));
  failed |= compare(label, "goertzel fma", fmaValue, goertzelReference(input, kLength, kBin, 1));
  if (floatBits(offValue) != floatBits(fmaValue))
    ++*split;
  free(off.allocated);
  free(fma.allocated);
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

int main(void) {
  int64_t split = 0;
  /* Full-mantissa samples against a full-mantissa coefficient make every
   * recursion product inexact, so off and fma are forced apart. */
  float directed[kLength] = {0};
  directed[0] = 0x1.3c6ef3p+0f;
  directed[1] = -0x1.1e2d5bp+0f;
  directed[2] = 0x1.7a4c9dp+0f;
  int failed = check(directed, "contract split", &split);

  float input[kLength];
  uint32_t state = UINT32_C(0x51A3C77D);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    snprintf(label, sizeof label, "trial %d", trial);
    for (int i = 0; i < kLength; ++i)
      input[i] = randomValue(&state);
    failed |= check(input, label, &split);
  }
  if (split == 0) {
    fprintf(stderr, "corpus is vacuous: off and fma agree everywhere\n");
    failed = 1;
  }
  return failed;
}
