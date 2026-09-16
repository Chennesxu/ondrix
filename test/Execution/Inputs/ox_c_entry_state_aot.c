#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { kLength = 16 };

// The language default contract of sos_df2_fixed at q15.
static const struct Policy policy = {
    .width = 16,
    .frac = 15,
    .accumulator_width = 40,
    .accumulator_frac = 30,
    .update_overflow = WRAP,
    .state_rounding = NEAREST_TIES_POSITIVE,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_TIES_POSITIVE,
    .output_overflow = SATURATE,
};

static void reference(const int16_t *input, const int16_t coefficients[5], int16_t scale,
                      const int16_t initial_state[2], int16_t *output, int16_t next_state[2]) {
  int64_t d1 = initial_state[0], d2 = initial_state[1];
  for (int index = 0; index < kLength; ++index) {
    int64_t state_accumulator = update_reference(0, input[index], scale, &policy);
    state_accumulator = update_reference(state_accumulator, d1, coefficients[3], &policy);
    state_accumulator = update_reference(state_accumulator, d2, coefficients[4], &policy);
    int64_t next_d1 =
        export_reference(state_accumulator, policy.state_rounding, policy.state_overflow, &policy);
    int64_t output_accumulator = update_reference(0, next_d1, coefficients[0], &policy);
    output_accumulator = update_reference(output_accumulator, d1, coefficients[1], &policy);
    output_accumulator = update_reference(output_accumulator, d2, coefficients[2], &policy);
    output[index] = (int16_t)export_reference(output_accumulator, policy.output_rounding,
                                              policy.output_overflow, &policy);
    d2 = d1;
    d1 = next_d1;
  }
  next_state[0] = (int16_t)d1;
  next_state[1] = (int16_t)d2;
}

static int differs(const char *label, const int16_t *actual, const int16_t *expected, int count) {
  for (int i = 0; i < count; ++i)
    if (actual[i] != expected[i]) {
      fprintf(stderr, "%s[%d]: got %d, expected %d\n", label, i, actual[i], expected[i]);
      return 1;
    }
  return 0;
}

int main(void) {
  const int16_t input[2][kLength] = {
      {1, -3, INT16_MAX, INT16_MIN, 16384, -8192, 12345, -12345, 7, 0, -1, 30000, -30000, 99, -99,
       5},
      {INT16_MIN, INT16_MAX, 3, -3, 1000, -1000, 250, -250, 0, 0, 1, -1, 20000, -20000, 8, -8}};
  const int16_t coefficients[5] = {INT16_MAX, 16384, 8192, 12288, -4096};
  const int16_t scales[1] = {INT16_MAX};
  int16_t state[2] = {3000, -5000}, saved[2], output[kLength], next[2];
  int16_t expectedOutput[kLength], expectedNext[2];
  // Two calls chained through the caller's own state buffers: the input
  // state must survive each call unchanged, and the next state must be the
  // reference's so the second call continues the same filter.
  for (int call = 0; call < 2; ++call) {
    memcpy(saved, state, sizeof(saved));
    reference(input[call], coefficients, scales[0], state, expectedOutput, expectedNext);
    ondrix_q15_sos_state_entry(input[call], coefficients, scales, state, output, next);
    if (differs("output", output, expectedOutput, kLength) ||
        differs("next_state", next, expectedNext, 2))
      return 1;
    if (memcmp(saved, state, sizeof(saved)) != 0) {
      fprintf(stderr, "call %d wrote into the const input state\n", call);
      return 1;
    }
    memcpy(state, next, sizeof(state));
  }
  return 0;
}
