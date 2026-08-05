#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MEMREF_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)
#define CALL_STREAM(function, input, input_size, coeffs, coeff_size, state, state_size, index)     \
  function(MEMREF_ARGS(input, input_size), MEMREF_ARGS(coeffs, coeff_size),                        \
           MEMREF_ARGS(state, state_size), index)

extern int16_t q15_stream_output_value_ties_positive(int16_t *, int16_t *, int64_t, int64_t,
                                                     int64_t, int16_t *, int16_t *, int64_t,
                                                     int64_t, int64_t, int16_t *, int16_t *,
                                                     int64_t, int64_t, int64_t, int64_t);
extern int16_t q15_stream_output_value(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                       int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                       int64_t, int64_t, int64_t, int64_t);
extern int16_t q15_stream_state_value(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                      int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                      int64_t, int64_t, int64_t, int64_t);
extern int32_t q31_stream_output_value(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                       int32_t *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
                                       int64_t, int64_t, int64_t, int64_t);
extern int32_t q31_stream_state_value(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                      int32_t *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
                                      int64_t, int64_t, int64_t, int64_t);
extern float f32_stream_output_value(float *, float *, int64_t, int64_t, int64_t, float *, float *,
                                     int64_t, int64_t, int64_t, float *, float *, int64_t, int64_t,
                                     int64_t, int64_t);
extern float f32_stream_state_value(float *, float *, int64_t, int64_t, int64_t, float *, float *,
                                    int64_t, int64_t, int64_t, float *, float *, int64_t, int64_t,
                                    int64_t, int64_t);
extern float f32_stream_off_output_value(float *, float *, int64_t, int64_t, int64_t, float *,
                                         float *, int64_t, int64_t, int64_t, float *, float *,
                                         int64_t, int64_t, int64_t, int64_t);

static int64_t clamp_i40(__int128 value) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  if (value < minimum)
    return (int64_t)minimum;
  if (value > maximum)
    return (int64_t)maximum;
  return (int64_t)value;
}

static int64_t clamp_i64(__int128 value) {
  if (value < INT64_MIN)
    return INT64_MIN;
  if (value > INT64_MAX)
    return INT64_MAX;
  return (int64_t)value;
}

// Independent ties-positive formulation: ITU-style add-half-then-floor-shift,
// total in __int128, deliberately not the compiler's quotient/remainder form.
static __int128 round_ties_positive(__int128 value, unsigned shift) {
  __int128 divisor = (__int128)1 << shift;
  __int128 shifted = value + (divisor >> 1);
  __int128 quotient = shifted / divisor;
  if (shifted % divisor < 0)
    --quotient;
  return quotient;
}

static __int128 round_nearest_even(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  __int128 remainder = value % divisor;
  if (remainder < 0) {
    remainder += divisor;
    --quotient;
  }
  const __int128 half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1) != 0))
    ++quotient;
  return quotient;
}

static void q15_reference(const int16_t *input, int64_t input_length, const int16_t *coefficients,
                          int64_t coefficient_length, const int16_t *state, int16_t *output,
                          int16_t *next_state, __int128 (*rounder)(__int128, unsigned)) {
  const int64_t history_length = coefficient_length - 1;
  for (int64_t sample = 0; sample < input_length; ++sample) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < coefficient_length; ++tap) {
      int64_t extended_index = sample + tap;
      int16_t value = extended_index < history_length ? state[extended_index]
                                                      : input[extended_index - history_length];
      accumulator = clamp_i40((__int128)accumulator + (__int128)value * coefficients[tap]);
    }
    __int128 result = rounder(accumulator, 15);
    if (result < INT16_MIN)
      result = INT16_MIN;
    if (result > INT16_MAX)
      result = INT16_MAX;
    output[sample] = (int16_t)result;
  }
  for (int64_t index = 0; index < history_length; ++index) {
    int64_t extended_index = input_length + index;
    next_state[index] = extended_index < history_length ? state[extended_index]
                                                        : input[extended_index - history_length];
  }
}

static void q31_reference(const int32_t *input, int64_t input_length, const int32_t *coefficients,
                          int64_t coefficient_length, const int32_t *state, int32_t *output,
                          int32_t *next_state) {
  const int64_t history_length = coefficient_length - 1;
  for (int64_t sample = 0; sample < input_length; ++sample) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < coefficient_length; ++tap) {
      int64_t extended_index = sample + tap;
      int32_t value = extended_index < history_length ? state[extended_index]
                                                      : input[extended_index - history_length];
      accumulator = clamp_i64((__int128)accumulator + (__int128)value * coefficients[tap]);
    }
    __int128 result = round_nearest_even(accumulator, 31);
    if (result < INT32_MIN)
      result = INT32_MIN;
    if (result > INT32_MAX)
      result = INT32_MAX;
    output[sample] = (int32_t)result;
  }
  for (int64_t index = 0; index < history_length; ++index) {
    int64_t extended_index = input_length + index;
    next_state[index] = extended_index < history_length ? state[extended_index]
                                                        : input[extended_index - history_length];
  }
}

static void f32_reference(const float *input, int64_t input_length, const float *coefficients,
                          int64_t coefficient_length, const float *state, float *output,
                          float *next_state) {
  const int64_t history_length = coefficient_length - 1;
  for (int64_t sample = 0; sample < input_length; ++sample) {
    float accumulator = 0.0f;
    for (int64_t tap = 0; tap < coefficient_length; ++tap) {
      int64_t extended_index = sample + tap;
      float value = extended_index < history_length ? state[extended_index]
                                                    : input[extended_index - history_length];
      accumulator = fmaf(value, coefficients[tap], accumulator);
    }
    output[sample] = accumulator;
  }
  for (int64_t index = 0; index < history_length; ++index) {
    int64_t extended_index = input_length + index;
    next_state[index] = extended_index < history_length ? state[extended_index]
                                                        : input[extended_index - history_length];
  }
}

static void f32_off_reference(const float *input, int64_t input_length, const float *coefficients,
                              int64_t coefficient_length, const float *state, float *output) {
  const int64_t history_length = coefficient_length - 1;
  for (int64_t sample = 0; sample < input_length; ++sample) {
    float accumulator = 0.0f;
    for (int64_t tap = 0; tap < coefficient_length; ++tap) {
      int64_t extended_index = sample + tap;
      float value = extended_index < history_length ? state[extended_index]
                                                    : input[extended_index - history_length];
      float product = value * coefficients[tap];
      accumulator = accumulator + product;
    }
    output[sample] = accumulator;
  }
}

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int check_q15(void) {
  int failed = 0;
  int16_t input[] = {32767, -32768, 4096, -8192, 16384, 1, -2, 3, -4};
  int16_t coefficients[] = {-32768, 16384, 8192, -4096, 2048};
  int16_t initial_state[] = {1000, -2000, 3000, -4000};
  int16_t expected_output[9], expected_state[4];
  q15_reference(input, 9, coefficients, 5, initial_state, expected_output, expected_state,
                round_nearest_even);

  for (int64_t index = 0; index < 9; ++index) {
    int16_t actual =
        CALL_STREAM(q15_stream_output_value, input, 9, coefficients, 5, initial_state, 4, index);
    if (actual != expected_output[index]) {
      fprintf(stderr, "Q15 whole output %lld: expected %d, got %d\n", (long long)index,
              expected_output[index], actual);
      failed = 1;
    }
  }
  for (int64_t index = 0; index < 4; ++index) {
    int16_t actual =
        CALL_STREAM(q15_stream_state_value, input, 9, coefficients, 5, initial_state, 4, index);
    if (actual != expected_state[index]) {
      fprintf(stderr, "Q15 whole state %lld: expected %d, got %d\n", (long long)index,
              expected_state[index], actual);
      failed = 1;
    }
  }

  int16_t first_output[2], split_state[4];
  q15_reference(input, 2, coefficients, 5, initial_state, first_output, split_state,
                round_nearest_even);
  for (int64_t index = 0; index < 2; ++index) {
    int16_t actual =
        CALL_STREAM(q15_stream_output_value, input, 2, coefficients, 5, initial_state, 4, index);
    if (actual != expected_output[index])
      failed = 1;
  }
  for (int64_t index = 0; index < 4; ++index)
    split_state[index] =
        CALL_STREAM(q15_stream_state_value, input, 2, coefficients, 5, initial_state, 4, index);
  for (int64_t index = 0; index < 7; ++index) {
    int16_t actual =
        CALL_STREAM(q15_stream_output_value, input + 2, 7, coefficients, 5, split_state, 4, index);
    if (actual != expected_output[index + 2]) {
      fprintf(stderr, "Q15 split output %lld: expected %d, got %d\n", (long long)index,
              expected_output[index + 2], actual);
      failed = 1;
    }
  }
  for (int64_t index = 0; index < 4; ++index) {
    int16_t actual =
        CALL_STREAM(q15_stream_state_value, input + 2, 7, coefficients, 5, split_state, 4, index);
    if (actual != expected_state[index])
      failed = 1;
  }

  int16_t dummy = 0;
  for (int64_t index = 0; index < 4; ++index) {
    int16_t actual =
        CALL_STREAM(q15_stream_state_value, &dummy, 0, coefficients, 5, initial_state, 4, index);
    if (actual != initial_state[index])
      failed = 1;
  }

  int16_t unit_coefficient[] = {16384};
  for (int64_t index = 0; index < 9; ++index) {
    int16_t actual =
        CALL_STREAM(q15_stream_output_value, input, 9, unit_coefficient, 1, &dummy, 0, index);
    int64_t accumulator = (int64_t)input[index] * unit_coefficient[0];
    int16_t direct = (int16_t)round_nearest_even(accumulator, 15);
    if (actual != direct)
      failed = 1;
  }

  // Ties-positive export witnesses: taps +-0.5 make each accumulator
  // 16384*(e[s]-e[s+1]) over the state-extended sequence, so odd differences
  // are exact half ties; even floor quotients diverge from nearest_even at
  // both signs, and difference 65533 lands the increment exactly on +32767.
  int16_t ntp_input[] = {1, 0, 3, 32767, -32766, 0, 0, 0};
  int16_t ntp_coefficients[] = {16384, -16384, 0, 0, 0};
  int16_t ntp_state[] = {0, 0, 0, 0};
  int16_t ntp_expected[8], ntp_even[8], ntp_next[4];
  q15_reference(ntp_input, 8, ntp_coefficients, 5, ntp_state, ntp_expected, ntp_next,
                round_ties_positive);
  q15_reference(ntp_input, 8, ntp_coefficients, 5, ntp_state, ntp_even, ntp_next,
                round_nearest_even);
  int ntp_divergent = 0;
  for (int64_t index = 0; index < 8; ++index) {
    int16_t actual = CALL_STREAM(q15_stream_output_value_ties_positive, ntp_input, 8,
                                 ntp_coefficients, 5, ntp_state, 4, index);
    if (actual != ntp_expected[index]) {
      fprintf(stderr, "Q15 ties-positive output %lld: expected %d, got %d\n", (long long)index,
              ntp_expected[index], actual);
      failed = 1;
    }
    if (ntp_expected[index] != ntp_even[index])
      ++ntp_divergent;
  }
  if (ntp_divergent != 3) {
    fprintf(stderr,
            "Q15 ties-positive corpus must diverge from nearest_even on 3 of 8 outputs, "
            "diverged on %d\n",
            ntp_divergent);
    failed = 1;
  }
  return failed;
}

static int check_q31(void) {
  int failed = 0;
  int32_t input[] = {INT32_MIN, INT32_MAX, INT32_C(1) << 30, -(INT32_C(1) << 30), 17};
  int32_t coefficients[] = {INT32_MIN, INT32_MAX, INT32_C(1) << 29};
  int32_t initial_state[] = {-31, 63};
  int32_t expected_output[5], expected_state[2];
  q31_reference(input, 5, coefficients, 3, initial_state, expected_output, expected_state);

  for (int64_t index = 0; index < 5; ++index) {
    int32_t actual =
        CALL_STREAM(q31_stream_output_value, input, 5, coefficients, 3, initial_state, 2, index);
    if (actual != expected_output[index])
      failed = 1;
  }
  for (int64_t index = 0; index < 2; ++index) {
    int32_t actual =
        CALL_STREAM(q31_stream_state_value, input, 5, coefficients, 3, initial_state, 2, index);
    if (actual != expected_state[index])
      failed = 1;
  }

  int32_t split_state[2];
  for (int64_t index = 0; index < 1; ++index) {
    int32_t actual =
        CALL_STREAM(q31_stream_output_value, input, 1, coefficients, 3, initial_state, 2, index);
    if (actual != expected_output[index])
      failed = 1;
  }
  for (int64_t index = 0; index < 2; ++index)
    split_state[index] =
        CALL_STREAM(q31_stream_state_value, input, 1, coefficients, 3, initial_state, 2, index);
  for (int64_t index = 0; index < 4; ++index) {
    int32_t actual =
        CALL_STREAM(q31_stream_output_value, input + 1, 4, coefficients, 3, split_state, 2, index);
    if (actual != expected_output[index + 1])
      failed = 1;
  }
  for (int64_t index = 0; index < 2; ++index) {
    int32_t actual =
        CALL_STREAM(q31_stream_state_value, input + 1, 4, coefficients, 3, split_state, 2, index);
    if (actual != expected_state[index])
      failed = 1;
  }
  return failed;
}

static int check_f32(void) {
  int failed = 0;
  float input[] = {-0.0f, 0x1.000002p+0f, -0.5f, 0.25f};
  float coefficients[] = {1.0f, 0x1.fffffcp-1f, -0.5f};
  float initial_state[] = {0.75f, -1.0f};
  float expected_output[4], expected_state[2];
  f32_reference(input, 4, coefficients, 3, initial_state, expected_output, expected_state);

  for (int64_t index = 0; index < 4; ++index) {
    float actual =
        CALL_STREAM(f32_stream_output_value, input, 4, coefficients, 3, initial_state, 2, index);
    if (f32_bits(actual) != f32_bits(expected_output[index]))
      failed = 1;
  }
  for (int64_t index = 0; index < 2; ++index) {
    float actual =
        CALL_STREAM(f32_stream_state_value, input, 4, coefficients, 3, initial_state, 2, index);
    if (f32_bits(actual) != f32_bits(expected_state[index]))
      failed = 1;
  }

  float split_state[2];
  float first_output[1];
  f32_reference(input, 1, coefficients, 3, initial_state, first_output, split_state);
  for (int64_t index = 0; index < 2; ++index)
    split_state[index] =
        CALL_STREAM(f32_stream_state_value, input, 1, coefficients, 3, initial_state, 2, index);
  for (int64_t index = 0; index < 1; ++index) {
    float actual =
        CALL_STREAM(f32_stream_output_value, input, 1, coefficients, 3, initial_state, 2, index);
    if (f32_bits(actual) != f32_bits(expected_output[index]))
      failed = 1;
  }
  for (int64_t index = 0; index < 3; ++index) {
    float actual =
        CALL_STREAM(f32_stream_output_value, input + 1, 3, coefficients, 3, split_state, 2, index);
    if (f32_bits(actual) != f32_bits(expected_output[index + 1]))
      failed = 1;
  }
  for (int64_t index = 0; index < 2; ++index) {
    float actual =
        CALL_STREAM(f32_stream_state_value, input + 1, 3, coefficients, 3, split_state, 2, index);
    if (f32_bits(actual) != f32_bits(expected_state[index]))
      failed = 1;
  }

  float off_input[] = {0x1.000002p+0f};
  float off_coefficients[] = {1.0f, 0x1.fffffcp-1f};
  float off_state[] = {-1.0f};
  float off_expected[1];
  f32_off_reference(off_input, 1, off_coefficients, 2, off_state, off_expected);
  float off_actual =
      CALL_STREAM(f32_stream_off_output_value, off_input, 1, off_coefficients, 2, off_state, 1, 0);
  if (f32_bits(off_expected[0]) != UINT32_C(0) ||
      f32_bits(off_actual) != f32_bits(off_expected[0])) {
    fprintf(stderr, "F32 off contract: expected 0x%08x, got 0x%08x\n", f32_bits(off_expected[0]),
            f32_bits(off_actual));
    failed = 1;
  }
  return failed;
}

int main(void) { return check_q15() | check_q31() | check_f32(); }
