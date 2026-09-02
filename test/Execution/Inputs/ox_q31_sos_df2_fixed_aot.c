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
} MemRefI32Rank1;

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRefI32Rank2;

struct SosResult {
  MemRefI32Rank1 output;
  MemRefI32Rank2 next_state;
};

struct Execution {
  struct SosResult result;
  int32_t *state_storage;
};

extern void _mlir_ciface_q31_sos_df2_fixed(struct SosResult *result, MemRefI32Rank1 *input,
                                           MemRefI32Rank2 *coefficients, MemRefI32Rank1 *scales,
                                           MemRefI32Rank2 *state);

static const struct Policy policy = {
    .width = 32,
    .frac = 31,
    .accumulator_width = 64,
    .accumulator_frac = 62,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = TOWARD_ZERO,
    .output_overflow = WRAP,
};

static int64_t clamped_updates = 0;

static int64_t update_counted(int64_t accumulator, int64_t lhs, int64_t rhs) {
  __int128 exact = (__int128)accumulator + (__int128)lhs * (__int128)rhs;
  int64_t updated = update_reference(accumulator, lhs, rhs, &policy);
  clamped_updates += exact != (__int128)updated;
  return updated;
}

static void reference(const int32_t *input, int64_t count, const int32_t coefficients[5],
                      int32_t scale, const int32_t initial_state[2], int32_t *output,
                      int32_t next_state[2]) {
  int64_t d1 = initial_state[0];
  int64_t d2 = initial_state[1];
  for (int64_t index = 0; index < count; ++index) {
    int64_t state_accumulator = update_counted(0, input[index], scale);
    state_accumulator = update_counted(state_accumulator, d1, coefficients[3]);
    state_accumulator = update_counted(state_accumulator, d2, coefficients[4]);
    int64_t next_d1 =
        export_reference(state_accumulator, policy.state_rounding, policy.state_overflow, &policy);

    int64_t output_accumulator = update_reference(0, next_d1, coefficients[0], &policy);
    output_accumulator = update_reference(output_accumulator, d1, coefficients[1], &policy);
    output_accumulator = update_reference(output_accumulator, d2, coefficients[2], &policy);
    output[index] = (int32_t)export_reference(output_accumulator, policy.output_rounding,
                                              policy.output_overflow, &policy);
    d2 = d1;
    d1 = next_d1;
  }
  next_state[0] = (int32_t)d1;
  next_state[1] = (int32_t)d2;
}

static struct Execution execute(const int32_t *input, int64_t count, const int32_t coefficients[5],
                                const int32_t scales[1], const int32_t state[2]) {
  int32_t *state_storage = malloc(2 * sizeof(int32_t));
  state_storage[0] = state[0];
  state_storage[1] = state[1];
  MemRefI32Rank1 input_ref = {(int32_t *)input, (int32_t *)input, 0, {count}, {1}};
  MemRefI32Rank2 coefficient_ref = {
      (int32_t *)coefficients, (int32_t *)coefficients, 0, {1, 5}, {5, 1}};
  MemRefI32Rank1 scale_ref = {(int32_t *)scales, (int32_t *)scales, 0, {1}, {1}};
  MemRefI32Rank2 state_ref = {state_storage, state_storage, 0, {1, 2}, {2, 1}};
  struct Execution execution = {.state_storage = state_storage};
  _mlir_ciface_q31_sos_df2_fixed(&execution.result, &input_ref, &coefficient_ref, &scale_ref,
                                 &state_ref);
  return execution;
}

static int check_output(const MemRefI32Rank1 *actual, const int32_t *expected, int64_t count,
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

static int check_state(const MemRefI32Rank2 *actual, const int32_t expected[2], const char *label) {
  if (actual->sizes[0] != 1 || actual->sizes[1] != 2)
    return 1;
  for (int64_t slot = 0; slot < 2; ++slot) {
    int32_t value = actual->aligned[actual->offset + slot * actual->strides[1]];
    if (value != expected[slot]) {
      fprintf(stderr, "%s[%lld]: got %d, expected %d\n", label, (long long)slot, value,
              expected[slot]);
      return 1;
    }
  }
  return 0;
}

static void copy_state(const MemRefI32Rank2 *source, int32_t destination[2]) {
  for (int64_t slot = 0; slot < 2; ++slot)
    destination[slot] = source->aligned[source->offset + slot * source->strides[1]];
}

static void release(struct Execution *execution) {
  free(execution->result.output.allocated);
  if (execution->result.next_state.allocated != execution->state_storage)
    free(execution->result.next_state.allocated);
  free(execution->state_storage);
}

int main(void) {
  /* Two exact 2^62 products (-1.0 * -1.0 on the input and the feedback) sum
   * to 2^63 inside the recurrence, so the saturating update is exercised too. */
  const int32_t input[7] = {INT32_MIN, -3, INT32_MAX, INT32_MIN, 1073741824, -536870912, 809041920};
  const int32_t coefficients[5] = {INT32_MAX, 1073741824, 536870912, INT32_MIN, INT32_MIN};
  const int32_t scales[1] = {INT32_MIN};
  const int32_t initial_state[2] = {INT32_MIN, INT32_MAX};
  int32_t expected_output[7];
  int32_t expected_state[2];
  reference(input, 7, coefficients, scales[0], initial_state, expected_output, expected_state);

  struct Execution whole = execute(input, 7, coefficients, scales, initial_state);
  int failed = check_output(&whole.result.output, expected_output, 7, "whole output") ||
               check_state(&whole.result.next_state, expected_state, "whole state");
  release(&whole);

  int32_t first_expected[3];
  int32_t split_state[2];
  reference(input, 3, coefficients, scales[0], initial_state, first_expected, split_state);
  struct Execution first = execute(input, 3, coefficients, scales, initial_state);
  failed |= check_output(&first.result.output, first_expected, 3, "first output");
  failed |= check_state(&first.result.next_state, split_state, "first state");
  copy_state(&first.result.next_state, split_state);

  int32_t second_expected[4];
  int32_t final_state[2];
  reference(input + 3, 4, coefficients, scales[0], split_state, second_expected, final_state);
  struct Execution second = execute(input + 3, 4, coefficients, scales, split_state);
  failed |= check_output(&second.result.output, second_expected, 4, "second output");
  failed |= check_state(&second.result.next_state, final_state, "second state");
  for (int64_t index = 0; index < 3; ++index)
    failed |= first_expected[index] != expected_output[index];
  for (int64_t index = 0; index < 4; ++index)
    failed |= second_expected[index] != expected_output[index + 3];
  for (int64_t slot = 0; slot < 2; ++slot)
    failed |= final_state[slot] != expected_state[slot];

  release(&first);
  release(&second);
  if (clamped_updates == 0) {
    fprintf(stderr, "corpus is vacuous: no update reached the i64 rail\n");
    failed = 1;
  }
  return failed;
}
