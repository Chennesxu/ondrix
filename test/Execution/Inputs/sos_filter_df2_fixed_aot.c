#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "fixed_point_reference.h"

#define MEMREF1(T) T *, T *, int64_t, int64_t, int64_t
#define MEMREF2(T) T *, T *, int64_t, int64_t, int64_t, int64_t, int64_t
#define MEMREF1_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)
#define MEMREF2_ARGS(pointer, rows, columns)                                                       \
  pointer, pointer, INT64_C(0), rows, columns, columns, INT64_C(1)

typedef int32_t (*Output16)(MEMREF1(int16_t), MEMREF2(int16_t), MEMREF1(int16_t), MEMREF2(int16_t),
                            int64_t);
typedef int32_t (*State16)(MEMREF1(int16_t), MEMREF2(int16_t), MEMREF1(int16_t), MEMREF2(int16_t),
                           int64_t, int64_t);
typedef int32_t (*Output32)(MEMREF1(int32_t), MEMREF2(int32_t), MEMREF1(int32_t), MEMREF2(int32_t),
                            int64_t);
typedef int32_t (*State32)(MEMREF1(int32_t), MEMREF2(int32_t), MEMREF1(int32_t), MEMREF2(int32_t),
                           int64_t, int64_t);

extern int32_t sos_fixed_q15_output_value(MEMREF1(int16_t), MEMREF2(int16_t), MEMREF1(int16_t),
                                          MEMREF2(int16_t), int64_t);
extern int32_t sos_fixed_q15_state_value(MEMREF1(int16_t), MEMREF2(int16_t), MEMREF1(int16_t),
                                         MEMREF2(int16_t), int64_t, int64_t);
extern int32_t sos_fixed_q31_output_value(MEMREF1(int32_t), MEMREF2(int32_t), MEMREF1(int32_t),
                                          MEMREF2(int32_t), int64_t);
extern int32_t sos_fixed_q31_state_value(MEMREF1(int32_t), MEMREF2(int32_t), MEMREF1(int32_t),
                                         MEMREF2(int32_t), int64_t, int64_t);
extern int32_t sos_fixed_q15_wrap_output_value(MEMREF1(int16_t), MEMREF2(int16_t), MEMREF1(int16_t),
                                               MEMREF2(int16_t), int64_t);
extern int32_t sos_fixed_q15_wrap_state_value(MEMREF1(int16_t), MEMREF2(int16_t), MEMREF1(int16_t),
                                              MEMREF2(int16_t), int64_t, int64_t);
extern int32_t sos_fixed_q31_saturate_output_value(MEMREF1(int32_t), MEMREF2(int32_t),
                                                   MEMREF1(int32_t), MEMREF2(int32_t), int64_t);
extern int32_t sos_fixed_q31_saturate_state_value(MEMREF1(int32_t), MEMREF2(int32_t),
                                                  MEMREF1(int32_t), MEMREF2(int32_t), int64_t,
                                                  int64_t);

static int64_t section_reference(int64_t input, const int64_t *coefficient, int64_t scale,
                                 int64_t *state, const struct Policy *policy) {
  int64_t old_d1 = state[0];
  int64_t old_d2 = state[1];
  int64_t state_accumulator = update_reference(0, input, scale, policy);
  state_accumulator = update_reference(state_accumulator, old_d1, coefficient[3], policy);
  state_accumulator = update_reference(state_accumulator, old_d2, coefficient[4], policy);
  int64_t next_d1 =
      export_reference(state_accumulator, policy->state_rounding, policy->state_overflow, policy);

  int64_t output_accumulator = update_reference(0, next_d1, coefficient[0], policy);
  output_accumulator = update_reference(output_accumulator, old_d1, coefficient[1], policy);
  output_accumulator = update_reference(output_accumulator, old_d2, coefficient[2], policy);
  int64_t output = export_reference(output_accumulator, policy->output_rounding,
                                    policy->output_overflow, policy);
  state[0] = next_d1;
  state[1] = old_d1;
  return output;
}

static void sos_reference(const int64_t *input, int64_t length, const int64_t *coefficients,
                          int64_t sections, const int64_t *scales, const int64_t *initial_state,
                          int64_t *output, int64_t *next_state, const struct Policy *policy) {
  memcpy(next_state, initial_state, (size_t)(sections * 2) * sizeof(int64_t));
  for (int64_t sample = 0; sample < length; ++sample) {
    int64_t value = input[sample];
    for (int64_t section = 0; section < sections; ++section)
      value = section_reference(value, coefficients + section * 5, scales[section],
                                next_state + section * 2, policy);
    output[sample] = value;
  }
}

static int32_t invoke_output16(Output16 function, int16_t *input, int64_t length,
                               int16_t *coefficients, int16_t *scales, const int16_t *state,
                               int64_t index) {
  int16_t state_copy[4];
  memcpy(state_copy, state, sizeof(state_copy));
  return function(MEMREF1_ARGS(input, length), MEMREF2_ARGS(coefficients, 2, 5),
                  MEMREF1_ARGS(scales, 2), MEMREF2_ARGS(state_copy, 2, 2), index);
}

static int32_t invoke_state16(State16 function, int16_t *input, int64_t length,
                              int16_t *coefficients, int16_t *scales, const int16_t *state,
                              int64_t section, int64_t slot) {
  int16_t state_copy[4];
  memcpy(state_copy, state, sizeof(state_copy));
  return function(MEMREF1_ARGS(input, length), MEMREF2_ARGS(coefficients, 2, 5),
                  MEMREF1_ARGS(scales, 2), MEMREF2_ARGS(state_copy, 2, 2), section, slot);
}

static int32_t invoke_output32(Output32 function, int32_t *input, int64_t length,
                               int32_t *coefficients, int32_t *scales, const int32_t *state,
                               int64_t index) {
  int32_t state_copy[4];
  memcpy(state_copy, state, sizeof(state_copy));
  return function(MEMREF1_ARGS(input, length), MEMREF2_ARGS(coefficients, 2, 5),
                  MEMREF1_ARGS(scales, 2), MEMREF2_ARGS(state_copy, 2, 2), index);
}

static int32_t invoke_state32(State32 function, int32_t *input, int64_t length,
                              int32_t *coefficients, int32_t *scales, const int32_t *state,
                              int64_t section, int64_t slot) {
  int32_t state_copy[4];
  memcpy(state_copy, state, sizeof(state_copy));
  return function(MEMREF1_ARGS(input, length), MEMREF2_ARGS(coefficients, 2, 5),
                  MEMREF1_ARGS(scales, 2), MEMREF2_ARGS(state_copy, 2, 2), section, slot);
}

static int check_q15(void) {
  int16_t input[] = {1, -3, INT16_MAX, INT16_MIN, 16384, -8192, 12345};
  int16_t coefficients[] = {INT16_MAX, 16384, 8192, 12288, -4096, 24576, -8192, 4096, 6144, -2048};
  int16_t scales[] = {INT16_MAX, 24576};
  int16_t initial_state[] = {3000, -5000, 7000, -9000};
  int64_t wide_input[7], wide_coefficients[10], wide_scales[2], wide_state[4];
  for (int i = 0; i < 7; ++i)
    wide_input[i] = input[i];
  for (int i = 0; i < 10; ++i)
    wide_coefficients[i] = coefficients[i];
  for (int i = 0; i < 2; ++i)
    wide_scales[i] = scales[i];
  for (int i = 0; i < 4; ++i)
    wide_state[i] = initial_state[i];
  const struct Policy policy = {16,           15,       40,          30,  SATURATE,
                                NEAREST_EVEN, SATURATE, TOWARD_ZERO, WRAP};
  int64_t expected_output[7], expected_state[4];
  sos_reference(wide_input, 7, wide_coefficients, 2, wide_scales, wide_state, expected_output,
                expected_state, &policy);

  int failed = 0;
  for (int64_t i = 0; i < 7; ++i) {
    int32_t actual = invoke_output16(sos_fixed_q15_output_value, input, 7, coefficients, scales,
                                     initial_state, i);
    if (actual != expected_output[i]) {
      fprintf(stderr, "Q15 output %lld: expected %lld, got %d\n", (long long)i,
              (long long)expected_output[i], actual);
      failed = 1;
    }
  }
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot) {
      int32_t actual = invoke_state16(sos_fixed_q15_state_value, input, 7, coefficients, scales,
                                      initial_state, section, slot);
      if (actual != expected_state[section * 2 + slot]) {
        fprintf(stderr, "Q15 state %lld,%lld mismatch\n", (long long)section, (long long)slot);
        failed = 1;
      }
    }

  int16_t split_state[4];
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      split_state[section * 2 + slot] = (int16_t)invoke_state16(
          sos_fixed_q15_state_value, input, 3, coefficients, scales, initial_state, section, slot);
  for (int64_t i = 0; i < 4; ++i)
    if (invoke_output16(sos_fixed_q15_output_value, input + 3, 4, coefficients, scales, split_state,
                        i) != expected_output[i + 3])
      failed = 1;
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      if (invoke_state16(sos_fixed_q15_state_value, input + 3, 4, coefficients, scales, split_state,
                         section, slot) != expected_state[section * 2 + slot]) {
        fprintf(stderr, "Q15 split state %lld,%lld mismatch\n", (long long)section,
                (long long)slot);
        failed = 1;
      }

  int16_t dummy = 0;
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      if (invoke_state16(sos_fixed_q15_state_value, &dummy, 0, coefficients, scales, initial_state,
                         section, slot) != initial_state[section * 2 + slot])
        failed = 1;
  return failed;
}

static int check_q31(void) {
  int32_t input[] = {INT32_MIN, 1, -3, INT32_MAX, 1073741824, -536870912};
  int32_t coefficients[] = {INT32_MAX,  1073741824, -536870912, INT32_MIN, INT32_MIN,
                            1610612736, -536870912, 268435456,  536870912, -268435456};
  int32_t scales[] = {INT32_MIN, 1610612736};
  int32_t initial_state[] = {INT32_MIN, INT32_MIN, 123456789, -345678901};
  int64_t wide_input[6], wide_coefficients[10], wide_scales[2], wide_state[4];
  for (int i = 0; i < 6; ++i)
    wide_input[i] = input[i];
  for (int i = 0; i < 10; ++i)
    wide_coefficients[i] = coefficients[i];
  for (int i = 0; i < 2; ++i)
    wide_scales[i] = scales[i];
  for (int i = 0; i < 4; ++i)
    wide_state[i] = initial_state[i];
  const struct Policy policy = {32, 31, 64, 62, WRAP, TOWARD_ZERO, WRAP, TOWARD_NEGATIVE, SATURATE};
  int64_t expected_output[6], expected_state[4];
  sos_reference(wide_input, 6, wide_coefficients, 2, wide_scales, wide_state, expected_output,
                expected_state, &policy);

  int failed = 0;
  for (int64_t i = 0; i < 6; ++i) {
    int32_t actual = invoke_output32(sos_fixed_q31_output_value, input, 6, coefficients, scales,
                                     initial_state, i);
    if (actual != expected_output[i]) {
      fprintf(stderr, "Q31 output %lld: expected %lld, got %d\n", (long long)i,
              (long long)expected_output[i], actual);
      failed = 1;
    }
  }
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot) {
      int32_t actual = invoke_state32(sos_fixed_q31_state_value, input, 6, coefficients, scales,
                                      initial_state, section, slot);
      if (actual != expected_state[section * 2 + slot]) {
        fprintf(stderr, "Q31 state %lld,%lld mismatch\n", (long long)section, (long long)slot);
        failed = 1;
      }
    }

  int32_t split_state[4];
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      split_state[section * 2 + slot] = invoke_state32(
          sos_fixed_q31_state_value, input, 2, coefficients, scales, initial_state, section, slot);
  for (int64_t i = 0; i < 4; ++i)
    if (invoke_output32(sos_fixed_q31_output_value, input + 2, 4, coefficients, scales, split_state,
                        i) != expected_output[i + 2])
      failed = 1;
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      if (invoke_state32(sos_fixed_q31_state_value, input + 2, 4, coefficients, scales, split_state,
                         section, slot) != expected_state[section * 2 + slot]) {
        fprintf(stderr, "Q31 split state %lld,%lld mismatch\n", (long long)section,
                (long long)slot);
        failed = 1;
      }

  struct Policy saturating = policy;
  saturating.update_overflow = SATURATE;
  int64_t alternative_output[6], alternative_state[4];
  sos_reference(wide_input, 6, wide_coefficients, 2, wide_scales, wide_state, alternative_output,
                alternative_state, &saturating);
  if (alternative_state[0] == expected_state[0] && alternative_output[0] == expected_output[0]) {
    fprintf(stderr, "Q31 corpus does not distinguish wrapping and saturating updates\n");
    failed = 1;
  }

  int32_t dummy = 0;
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot)
      if (invoke_state32(sos_fixed_q31_state_value, &dummy, 0, coefficients, scales, initial_state,
                         section, slot) != initial_state[section * 2 + slot])
        failed = 1;
  return failed;
}

static int check_q15_impulse_golden(void) {
  int16_t input[] = {16384, 0, 0};
  int16_t coefficients[] = {16384, 8192, 4096, 12288, -4096, 24576, -8192, 4096, 6144, -2048};
  int16_t scales[] = {32767, 24576};
  int16_t initial_state[] = {0, 0, 0, 0};
  static const int32_t expected_output[] = {4608, 3360, 1854};
  static const int32_t expected_state[] = {256, 6144, 3624, 6528};
  int failed = 0;
  for (int64_t i = 0; i < 3; ++i) {
    int32_t actual = invoke_output16(sos_fixed_q15_output_value, input, 3, coefficients, scales,
                                     initial_state, i);
    if (actual != expected_output[i]) {
      fprintf(stderr, "Q15 impulse output %lld: expected %d, got %d\n", (long long)i,
              expected_output[i], actual);
      failed = 1;
    }
  }
  for (int64_t section = 0; section < 2; ++section)
    for (int64_t slot = 0; slot < 2; ++slot) {
      int32_t actual = invoke_state16(sos_fixed_q15_state_value, input, 3, coefficients, scales,
                                      initial_state, section, slot);
      int32_t expected = expected_state[section * 2 + slot];
      if (actual != expected) {
        fprintf(stderr, "Q15 impulse state %lld,%lld: expected %d, got %d\n", (long long)section,
                (long long)slot, expected, actual);
        failed = 1;
      }
    }
  return failed;
}

static int check_complementary_policies(void) {
  int16_t q15_input[] = {INT16_MIN, INT16_MAX, -3};
  int16_t q15_coefficients[] = {INT16_MAX, INT16_MAX, INT16_MAX, INT16_MAX, INT16_MIN,
                                INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX};
  int16_t q15_scales[] = {INT16_MIN, INT16_MAX};
  int16_t q15_state[] = {INT16_MAX, INT16_MIN, -12345, 23456};
  int64_t q15_wide_input[3], q15_wide_coefficients[10], q15_wide_scales[2], q15_wide_state[4];
  for (int i = 0; i < 3; ++i)
    q15_wide_input[i] = q15_input[i];
  for (int i = 0; i < 10; ++i)
    q15_wide_coefficients[i] = q15_coefficients[i];
  for (int i = 0; i < 2; ++i)
    q15_wide_scales[i] = q15_scales[i];
  for (int i = 0; i < 4; ++i)
    q15_wide_state[i] = q15_state[i];
  const struct Policy q15_policy = {16,   15,           40,      30, WRAP, TOWARD_NEGATIVE,
                                    WRAP, NEAREST_EVEN, SATURATE};
  int64_t q15_expected[3], q15_next_state[4];
  sos_reference(q15_wide_input, 3, q15_wide_coefficients, 2, q15_wide_scales, q15_wide_state,
                q15_expected, q15_next_state, &q15_policy);

  int32_t q31_input[] = {INT32_MIN, INT32_MAX, -3};
  int32_t q31_coefficients[] = {INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN, INT32_MIN,
                                INT32_MAX, INT32_MIN, INT32_MAX, INT32_MIN, INT32_MAX};
  int32_t q31_scales[] = {INT32_MIN, INT32_MAX};
  int32_t q31_state[] = {INT32_MIN, INT32_MIN, -123456789, 987654321};
  int64_t q31_wide_input[3], q31_wide_coefficients[10], q31_wide_scales[2], q31_wide_state[4];
  for (int i = 0; i < 3; ++i)
    q31_wide_input[i] = q31_input[i];
  for (int i = 0; i < 10; ++i)
    q31_wide_coefficients[i] = q31_coefficients[i];
  for (int i = 0; i < 2; ++i)
    q31_wide_scales[i] = q31_scales[i];
  for (int i = 0; i < 4; ++i)
    q31_wide_state[i] = q31_state[i];
  const struct Policy q31_policy = {32,           31,       64,          62,  SATURATE,
                                    NEAREST_EVEN, SATURATE, TOWARD_ZERO, WRAP};
  int64_t q31_expected[3], q31_next_state[4];
  sos_reference(q31_wide_input, 3, q31_wide_coefficients, 2, q31_wide_scales, q31_wide_state,
                q31_expected, q31_next_state, &q31_policy);

  int failed = 0;
  for (int64_t i = 0; i < 3; ++i) {
    int32_t q15_actual = invoke_output16(sos_fixed_q15_wrap_output_value, q15_input, 3,
                                         q15_coefficients, q15_scales, q15_state, i);
    int32_t q31_actual = invoke_output32(sos_fixed_q31_saturate_output_value, q31_input, 3,
                                         q31_coefficients, q31_scales, q31_state, i);
    if (q15_actual != q15_expected[i]) {
      fprintf(stderr, "Q15 complementary policy output %lld mismatch\n", (long long)i);
      failed = 1;
    }
    if (q31_actual != q31_expected[i]) {
      fprintf(stderr, "Q31 complementary policy output %lld mismatch\n", (long long)i);
      failed = 1;
    }
  }
  int32_t q15_final_state = invoke_state16(sos_fixed_q15_wrap_state_value, q15_input, 3,
                                           q15_coefficients, q15_scales, q15_state, 1, 0);
  if (q15_final_state != q15_next_state[2]) {
    fprintf(stderr, "Q15 complementary policy final state mismatch\n");
    failed = 1;
  }
  int32_t q31_final_state = invoke_state32(sos_fixed_q31_saturate_state_value, q31_input, 3,
                                           q31_coefficients, q31_scales, q31_state, 1, 0);
  if (q31_final_state != q31_next_state[2]) {
    fprintf(stderr, "Q31 complementary policy final state mismatch\n");
    failed = 1;
  }
  return failed;
}

int main(void) {
  return check_q15() | check_q31() | check_q15_impulse_golden() | check_complementary_policies();
}
