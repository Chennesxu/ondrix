#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

struct StreamResult {
  MemRefI32 output;
  MemRefI32 next_state;
};

extern void _mlir_ciface_q31_fir_stream(struct StreamResult *result, MemRefI32 *input,
                                        MemRefI32 *coefficients, MemRefI32 *state);

static const struct Policy policy = {
    .width = 32,
    .frac = 31,
    .accumulator_width = 64,
    .accumulator_frac = 62,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

static int64_t clamped_updates = 0;

static void stream_reference(const int32_t *input, int64_t input_count,
                             const int32_t coefficients[3], const int32_t state[2], int32_t *output,
                             int32_t next_state[2]) {
  for (int64_t n = 0; n < input_count; ++n) {
    int64_t accumulator = 0;
    for (int64_t k = 0; k < 3; ++k) {
      int64_t extended_index = n + k;
      int32_t sample = extended_index < 2 ? state[extended_index] : input[extended_index - 2];
      __int128 exact = (__int128)accumulator + (__int128)sample * (__int128)coefficients[k];
      accumulator = update_reference(accumulator, sample, coefficients[k], &policy);
      clamped_updates += exact != (__int128)accumulator;
    }
    output[n] = (int32_t)export_reference(accumulator, policy.output_rounding,
                                          policy.output_overflow, &policy);
  }

  for (int64_t index = 0; index < 2; ++index) {
    int64_t extended_index = input_count + index;
    next_state[index] = extended_index < 2 ? state[extended_index] : input[extended_index - 2];
  }
}

static struct StreamResult execute(const int32_t *input, int64_t input_count,
                                   const int32_t coefficients[3], const int32_t state[2]) {
  MemRefI32 input_ref = {(int32_t *)input, (int32_t *)input, 0, {input_count}, {1}};
  MemRefI32 coefficient_ref = {(int32_t *)coefficients, (int32_t *)coefficients, 0, {3}, {1}};
  MemRefI32 state_ref = {(int32_t *)state, (int32_t *)state, 0, {2}, {1}};
  struct StreamResult result;
  _mlir_ciface_q31_fir_stream(&result, &input_ref, &coefficient_ref, &state_ref);
  return result;
}

static int check_values(const MemRefI32 *actual, const int32_t *expected, int64_t count,
                        const char *label) {
  if (actual->sizes[0] != count)
    return 1;
  for (int64_t index = 0; index < count; ++index) {
    int32_t value = actual->aligned[actual->offset + index * actual->strides[0]];
    if (value != expected[index]) {
      fprintf(stderr, "%s[%lld]: got %d, expected %d\n", label, (long long)index, value,
              expected[index]);
      return 1;
    }
  }
  return 0;
}

int main(void) {
  /* Two full-scale Q31 products already reach the i64 rail, so this corpus
   * exercises the declared saturating update, not only the export. */
  const int32_t input[5] = {INT32_MIN, INT32_MAX, 1073741824, -536870912, 268435456};
  const int32_t coefficients[3] = {INT32_MIN, INT32_MAX, 1073741824};
  const int32_t initial_state[2] = {INT32_MIN, -1234567890};
  int32_t expected_output[5];
  int32_t expected_state[2];
  stream_reference(input, 5, coefficients, initial_state, expected_output, expected_state);

  struct StreamResult whole = execute(input, 5, coefficients, initial_state);
  int failed = check_values(&whole.output, expected_output, 5, "whole output") ||
               check_values(&whole.next_state, expected_state, 2, "whole state");
  free(whole.output.allocated);
  free(whole.next_state.allocated);

  int32_t first_expected[2];
  int32_t split_state[2];
  stream_reference(input, 2, coefficients, initial_state, first_expected, split_state);
  struct StreamResult first = execute(input, 2, coefficients, initial_state);
  failed |= check_values(&first.output, first_expected, 2, "first output");
  failed |= check_values(&first.next_state, split_state, 2, "first state");

  int32_t second_expected[3];
  int32_t final_state[2];
  stream_reference(input + 2, 3, coefficients, split_state, second_expected, final_state);
  struct StreamResult second = execute(input + 2, 3, coefficients, split_state);
  failed |= check_values(&second.output, second_expected, 3, "second output");
  failed |= check_values(&second.next_state, final_state, 2, "second state");
  for (int index = 0; index < 2; ++index)
    failed |= first_expected[index] != expected_output[index];
  for (int index = 0; index < 3; ++index)
    failed |= second_expected[index] != expected_output[index + 2];
  for (int index = 0; index < 2; ++index)
    failed |= final_state[index] != expected_state[index];

  free(first.output.allocated);
  free(first.next_state.allocated);
  free(second.output.allocated);
  free(second.next_state.allocated);
  if (clamped_updates == 0) {
    fprintf(stderr, "corpus is vacuous: no update reached the i64 rail\n");
    failed = 1;
  }
  return failed;
}
