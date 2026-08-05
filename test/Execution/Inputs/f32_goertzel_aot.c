#include "goertzel_f32_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Object gate for the f32 Goertzel contract. off and fma are exact, so those
 * are bit for bit against a reference that derives the same coefficient and
 * runs the same event graph. The fast contract's legal set still has two
 * members here, but the lowering selects the fused one and emits it
 * unflagged, so the object must equal the fma object bit for bit rather than
 * merely belong to a set. */

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
  failed |= compare(label, "goertzel off", offValue, goertzelReference(input, kLength, kBin, 0));
  failed |= compare(label, "goertzel fma", fmaValue, goertzelReference(input, kLength, kBin, 1));
  /* The fast contract's legal set still has two members, but the lowering
   * selects the fused one, so this pins the chosen graph rather than
   * membership. Restoring the declaration on the emitted event reddens it:
   * the backend de-fuses a reassoc-flagged fma and the object drops to the
   * off value. */
  failed |= compare(label, "goertzel fast", fastValue, fmaValue);
  failed |= compare(label, "goertzel quarter turn", quarter.aligned[quarter.offset],
                    goertzelReference(input, kLength, kQuarterTurnBin, 0));
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
