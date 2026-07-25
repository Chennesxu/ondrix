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

struct StreamResult {
  MemRefI16 output;
  MemRefI16 next_state;
};

extern void _mlir_ciface_q15_fir_stream(struct StreamResult *result, MemRefI16 *input,
                                        MemRefI16 *coefficients, MemRefI16 *state);

static const struct Policy policy = {
    .width = 16,
    .frac = 15,
    .accumulator_width = 33,
    .accumulator_frac = 30,
    .update_overflow = WRAP,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

static void stream_reference(const int16_t *input, int64_t input_count,
                             const int16_t coefficients[3], const int16_t state[2], int16_t *output,
                             int16_t next_state[2]) {
  for (int64_t n = 0; n < input_count; ++n) {
    int64_t accumulator = 0;
    for (int64_t k = 0; k < 3; ++k) {
      int64_t extended_index = n + k;
      int16_t sample = extended_index < 2 ? state[extended_index] : input[extended_index - 2];
      accumulator = update_reference(accumulator, sample, coefficients[k], &policy);
    }
    output[n] = (int16_t)export_reference(accumulator, policy.output_rounding,
                                          policy.output_overflow, &policy);
  }

  for (int64_t index = 0; index < 2; ++index) {
    int64_t extended_index = input_count + index;
    next_state[index] = extended_index < 2 ? state[extended_index] : input[extended_index - 2];
  }
}

static struct StreamResult execute(const int16_t *input, int64_t input_count,
                                   const int16_t coefficients[3], const int16_t state[2]) {
  MemRefI16 input_ref = {(int16_t *)input, (int16_t *)input, 0, {input_count}, {1}};
  MemRefI16 coefficient_ref = {(int16_t *)coefficients, (int16_t *)coefficients, 0, {3}, {1}};
  MemRefI16 state_ref = {(int16_t *)state, (int16_t *)state, 0, {2}, {1}};
  struct StreamResult result;
  _mlir_ciface_q15_fir_stream(&result, &input_ref, &coefficient_ref, &state_ref);
  return result;
}

static int check_values(const MemRefI16 *actual, const int16_t *expected, int64_t count,
                        const char *label) {
  if (actual->sizes[0] != count)
    return 1;
  for (int64_t index = 0; index < count; ++index) {
    int16_t value = actual->aligned[actual->offset + index * actual->strides[0]];
    if (value != expected[index]) {
      fprintf(stderr, "%s[%lld]: got %d, expected %d\n", label, (long long)index, value,
              expected[index]);
      return 1;
    }
  }
  return 0;
}

int main(void) {
  const int16_t input[5] = {-32768, 32767, 16384, -8192, 4096};
  const int16_t coefficients[3] = {16384, -8192, 4096};
  const int16_t initial_state[2] = {12345, -23456};
  int16_t expected_output[5];
  int16_t expected_state[2];
  stream_reference(input, 5, coefficients, initial_state, expected_output, expected_state);

  struct StreamResult whole = execute(input, 5, coefficients, initial_state);
  int failed = check_values(&whole.output, expected_output, 5, "whole output") ||
               check_values(&whole.next_state, expected_state, 2, "whole state");
  free(whole.output.allocated);
  free(whole.next_state.allocated);

  int16_t first_expected[2];
  int16_t split_state[2];
  stream_reference(input, 2, coefficients, initial_state, first_expected, split_state);
  struct StreamResult first = execute(input, 2, coefficients, initial_state);
  failed |= check_values(&first.output, first_expected, 2, "first output");
  failed |= check_values(&first.next_state, split_state, 2, "first state");

  int16_t second_expected[3];
  int16_t final_state[2];
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
  return failed;
}
