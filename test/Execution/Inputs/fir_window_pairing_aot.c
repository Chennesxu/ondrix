#include "fixed_point_reference.h"

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

extern void _mlir_ciface_fir_pairs_loop_odd(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fir_pairs_unrolled_odd(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fir_pairs_even(MemRefI16 *, MemRefI16 *, MemRefI16 *);

static const struct Policy policy = {
    .width = 16,
    .frac = 15,
    .accumulator_width = 40,
    .accumulator_frac = 30,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_TIES_POSITIVE,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_TIES_POSITIVE,
    .output_overflow = SATURATE,
};

typedef void (*Kernel)(MemRefI16 *, MemRefI16 *, MemRefI16 *);

static int failures;
static int64_t exportRails, negatives;

/* The ordered reference: output n folds taps 0..K-1 over input[n .. n+K) in
 * increasing tap order and exports once. No padded zero step exists here. */
static int16_t reference(const int16_t *input, const int16_t *taps, int64_t k, int64_t n) {
  int64_t accumulator = 0;
  for (int64_t j = 0; j < k; ++j)
    accumulator = update_reference(accumulator, input[n + j], taps[j], &policy);
  int64_t exported =
      export_reference(accumulator, policy.output_rounding, policy.output_overflow, &policy);
  int64_t raw = accumulator >> 15;
  if (raw > 32767 || raw < -32768)
    ++exportRails;
  if (exported < 0)
    ++negatives;
  return (int16_t)exported;
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static void check(const char *label, Kernel kernel, int64_t length, int64_t k, uint32_t seed) {
  int16_t input[64];
  int16_t taps[32];
  for (int trial = 0; trial < 8; ++trial) {
    uint32_t state = seed + (uint32_t)trial * 0x9E3779B9u;
    for (int64_t i = 0; i < length; ++i) {
      state = nextState(state);
      input[i] = trial == 0 ? (int16_t)(i % 2 == 0 ? INT16_MIN : INT16_MAX) : (int16_t)state;
    }
    for (int64_t j = 0; j < k; ++j) {
      state = nextState(state);
      taps[j] = trial == 0 ? INT16_MIN : (int16_t)state;
    }
    MemRefI16 inputRef = {input, input, 0, {length}, {1}};
    MemRefI16 tapRef = {taps, taps, 0, {k}, {1}};
    MemRefI16 got;
    kernel(&got, &inputRef, &tapRef);
    int64_t outputs = length - k + 1;
    if (got.sizes[0] != outputs) {
      fprintf(stderr, "%s: %lld outputs, expected %lld\n", label, (long long)got.sizes[0],
              (long long)outputs);
      ++failures;
    }
    for (int64_t n = 0; n < outputs; ++n) {
      int16_t expected = reference(input, taps, k, n);
      int16_t actual = got.aligned[got.offset + n * got.strides[0]];
      if (actual != expected && failures++ < 8)
        fprintf(stderr, "%s trial %d output %lld: got %d, expected %d\n", label, trial,
                (long long)n, actual, expected);
    }
    free(got.allocated);
  }
}

int main(void) {
  check("loop_odd", _mlir_ciface_fir_pairs_loop_odd, 64, 32, 0x4D9A3E17u);
  check("unrolled_odd", _mlir_ciface_fir_pairs_unrolled_odd, 12, 4, 0x2C90D1B3u);
  check("even", _mlir_ciface_fir_pairs_even, 13, 4, 0x6F0A9C27u);
  if (exportRails == 0 || negatives == 0) {
    fprintf(stderr, "corpus is vacuous: %lld export rails, %lld negatives\n",
            (long long)exportRails, (long long)negatives);
    ++failures;
  }
  return failures != 0;
}
