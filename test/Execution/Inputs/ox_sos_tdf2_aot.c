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

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRefF32Rank2;

struct SosResult {
  MemRefF32Rank1 output;
  MemRefF32Rank2 next_state;
};

extern void _mlir_ciface_f32_sos_tdf2(struct SosResult *, MemRefF32Rank1 *, MemRefF32Rank2 *,
                                      MemRefF32Rank1 *, MemRefF32Rank2 *);

enum { kSections = 2, kLength = 12, kSplit = 5 };

/* The declared TDF-II event graph under contract=off: every multiply-add is
 * the stated separate multiply and add, and the state updates are not
 * regrouped. Written from the operation contract, not from the lowering. */
static void reference(const float *input, int64_t length, const float coefficients[kSections][5],
                      const float scales[kSections], const float initial[kSections][2],
                      float *output, float next[kSections][2]) {
  memcpy(next, initial, sizeof(float) * kSections * 2);
  for (int64_t sample = 0; sample < length; ++sample) {
    float value = input[sample];
    for (int64_t section = 0; section < kSections; ++section) {
      const float *c = coefficients[section];
      float scaled = value * scales[section];
      float sectionOutput = next[section][0] + scaled * c[0];
      float feedback1 = sectionOutput * c[3];
      float firstTerm = feedback1 + scaled * c[1];
      float nextZ1 = next[section][1] + firstTerm;
      float feedback2 = sectionOutput * c[4];
      float nextZ2 = feedback2 + scaled * c[2];
      next[section][0] = nextZ1;
      next[section][1] = nextZ2;
      value = sectionOutput;
    }
    output[sample] = value;
  }
}

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct Execution {
  struct SosResult result;
  float *stateStorage;
};

static struct Execution execute(const float *input, int64_t length,
                                float coefficients[kSections][5], float scales[kSections],
                                const float state[kSections][2]) {
  /* Bufferization may update the state buffer in place, so every call gets a
   * fresh copy: harness ownership, not source tensor semantics. */
  float *stateStorage = malloc(sizeof(float) * kSections * 2);
  memcpy(stateStorage, state, sizeof(float) * kSections * 2);
  MemRefF32Rank1 inputRef = {(float *)input, (float *)input, 0, {length}, {1}};
  MemRefF32Rank2 coefficientRef = {
      &coefficients[0][0], &coefficients[0][0], 0, {kSections, 5}, {5, 1}};
  MemRefF32Rank1 scaleRef = {scales, scales, 0, {kSections}, {1}};
  MemRefF32Rank2 stateRef = {stateStorage, stateStorage, 0, {kSections, 2}, {2, 1}};
  struct Execution execution = {.stateStorage = stateStorage};
  _mlir_ciface_f32_sos_tdf2(&execution.result, &inputRef, &coefficientRef, &scaleRef, &stateRef);
  return execution;
}

static void release(struct Execution *execution) {
  free(execution->result.output.allocated);
  if (execution->result.next_state.allocated != execution->stateStorage)
    free(execution->result.next_state.allocated);
  free(execution->stateStorage);
}

static int checkOutput(const MemRefF32Rank1 *actual, const float *expected, int64_t length,
                       const char *label) {
  if (actual->sizes[0] != length)
    return 1;
  int failed = 0;
  for (int64_t index = 0; index < length; ++index) {
    float value = actual->aligned[actual->offset + index * actual->strides[0]];
    if (floatBits(value) != floatBits(expected[index])) {
      fprintf(stderr, "%s[%lld]: got %a, expected %a\n", label, (long long)index, (double)value,
              (double)expected[index]);
      failed = 1;
    }
  }
  return failed;
}

static int checkState(const MemRefF32Rank2 *actual, const float expected[kSections][2],
                      const char *label) {
  if (actual->sizes[0] != kSections || actual->sizes[1] != 2)
    return 1;
  int failed = 0;
  for (int64_t section = 0; section < kSections; ++section)
    for (int64_t slot = 0; slot < 2; ++slot) {
      float value =
          actual
              ->aligned[actual->offset + section * actual->strides[0] + slot * actual->strides[1]];
      if (floatBits(value) != floatBits(expected[section][slot])) {
        fprintf(stderr, "%s[%lld][%lld]: got %a, expected %a\n", label, (long long)section,
                (long long)slot, (double)value, (double)expected[section][slot]);
        failed = 1;
      }
    }
  return failed;
}

static void copyState(const MemRefF32Rank2 *source, float destination[kSections][2]) {
  for (int64_t section = 0; section < kSections; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      destination[section][slot] =
          source
              ->aligned[source->offset + section * source->strides[0] + slot * source->strides[1]];
}

int main(void) {
  float coefficients[kSections][5] = {
      {0x1.3c6ef3p-1f, -0x1.1e2d5bp-2f, 0x1.7a4c9dp-3f, -0x1.05a3c1p-1f, 0x1.9d2b7fp-3f},
      {0x1.921fb5p-1f, 0x1.4p-3f, -0x1.8p-4f, 0x1.3333p-2f, -0x1.c28f5cp-3f}};
  float scales[kSections] = {0x1.6a09e6p-1f, 0x1.0p+0f};
  const float initial[kSections][2] = {{0x1.8p-3f, -0x1.4p-4f}, {-0x1.2p-2f, 0x1.1p-5f}};
  float input[kLength];
  for (int64_t index = 0; index < kLength; ++index)
    input[index] = (float)((index * 7) % 11 - 5) * 0x1.3p-3f;

  float expectedOutput[kLength];
  float expectedState[kSections][2];
  reference(input, kLength, coefficients, scales, initial, expectedOutput, expectedState);

  struct Execution whole = execute(input, kLength, coefficients, scales, initial);
  int failed = checkOutput(&whole.result.output, expectedOutput, kLength, "whole output") ||
               checkState(&whole.result.next_state, expectedState, "whole state");
  release(&whole);

  /* Splitting the chunk must be bitwise identical: the state is the only
   * carrier between chunks. */
  float firstExpected[kSplit];
  float splitState[kSections][2];
  reference(input, kSplit, coefficients, scales, initial, firstExpected, splitState);
  struct Execution first = execute(input, kSplit, coefficients, scales, initial);
  failed |= checkOutput(&first.result.output, firstExpected, kSplit, "first output");
  failed |= checkState(&first.result.next_state, splitState, "first state");
  copyState(&first.result.next_state, splitState);
  release(&first);

  float secondExpected[kLength - kSplit];
  float finalState[kSections][2];
  reference(input + kSplit, kLength - kSplit, coefficients, scales, splitState, secondExpected,
            finalState);
  struct Execution second =
      execute(input + kSplit, kLength - kSplit, coefficients, scales, splitState);
  failed |= checkOutput(&second.result.output, secondExpected, kLength - kSplit, "second output");
  failed |= checkState(&second.result.next_state, finalState, "second state");
  release(&second);
  for (int64_t index = 0; index < kLength - kSplit; ++index)
    failed |= floatBits(secondExpected[index]) != floatBits(expectedOutput[index + kSplit]);

  /* Empty input returns an empty output and the unchanged state. */
  struct Execution empty = execute(input, 0, coefficients, scales, initial);
  failed |= empty.result.output.sizes[0] != 0;
  failed |= checkState(&empty.result.next_state, initial, "empty state");
  release(&empty);

  /* Non-vacuity: a recurrence that never left its initial state, or whose
   * output equalled its input, would gate nothing about the event graph. */
  int64_t moved = 0;
  for (int64_t index = 0; index < kLength; ++index)
    if (floatBits(expectedOutput[index]) != floatBits(input[index]))
      ++moved;
  if (moved != kLength || floatBits(expectedState[0][0]) == floatBits(initial[0][0])) {
    fprintf(stderr, "corpus is vacuous: %lld of %d samples moved\n", (long long)moved, kLength);
    failed = 1;
  }
  return failed;
}
