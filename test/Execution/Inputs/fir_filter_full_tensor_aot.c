#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MEMREF_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)

extern int16_t q15_full_filter_value(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                     int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                     int64_t, int64_t, int64_t, int64_t);
extern int32_t q31_full_filter_value(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                     int32_t *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
                                     int64_t, int64_t, int64_t, int64_t);
extern int16_t q15_full_filter_short_input(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                           int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                           int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                           int64_t);
extern float f32_full_filter_value(float *, float *, int64_t, int64_t, int64_t, float *, float *,
                                   int64_t, int64_t, int64_t, float *, float *, int64_t, int64_t,
                                   int64_t, int64_t);
extern int16_t q15_full_shared_coeff_init(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                          int16_t *, int16_t *, int64_t, int64_t, int64_t, int64_t);
extern int16_t q15_full_shared_input_init(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                          int16_t *, int16_t *, int64_t, int64_t, int64_t, int64_t);

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

static int16_t q15_reference(const int16_t *input, int64_t input_length,
                             const int16_t *coefficients, int64_t coefficient_length,
                             int64_t output_index) {
  int64_t accumulator = 0;
  const int64_t left_padding = coefficient_length - 1;
  for (int64_t tap = 0; tap < coefficient_length; ++tap) {
    const int64_t input_index = output_index + tap - left_padding;
    if (input_index < 0 || input_index >= input_length)
      continue;
    accumulator =
        clamp_i40((__int128)accumulator + (__int128)input[input_index] * coefficients[tap]);
  }
  __int128 result = round_nearest_even(accumulator, 15);
  if (result < INT16_MIN)
    return INT16_MIN;
  if (result > INT16_MAX)
    return INT16_MAX;
  return (int16_t)result;
}

static int32_t q31_reference(const int32_t *input, int64_t input_length,
                             const int32_t *coefficients, int64_t coefficient_length,
                             int64_t output_index) {
  int64_t accumulator = 0;
  const int64_t left_padding = coefficient_length - 1;
  for (int64_t tap = 0; tap < coefficient_length; ++tap) {
    const int64_t input_index = output_index + tap - left_padding;
    if (input_index < 0 || input_index >= input_length)
      continue;
    accumulator =
        clamp_i64((__int128)accumulator + (__int128)input[input_index] * coefficients[tap]);
  }
  __int128 result = round_nearest_even(accumulator, 31);
  if (result < INT32_MIN)
    return INT32_MIN;
  if (result > INT32_MAX)
    return INT32_MAX;
  return (int32_t)result;
}

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int main(void) {
  int failed = 0;

  int16_t q15_input[] = {32767, -32768, 4096, -8192, 16384, 1, -2, 3};
  int16_t q15_coefficients[] = {-32768, 16384, 8192, -4096, 2048};
  int16_t q15_output[12] = {0};
  for (int64_t index = 0; index < 12; ++index) {
    int16_t actual =
        q15_full_filter_value(MEMREF_ARGS(q15_input, 8), MEMREF_ARGS(q15_coefficients, 5),
                              MEMREF_ARGS(q15_output, 12), index);
    int16_t expected = q15_reference(q15_input, 8, q15_coefficients, 5, index);
    if (actual != expected) {
      fprintf(stderr, "Q15 full output %lld: expected %d, got %d\n", (long long)index, expected,
              actual);
      failed = 1;
    }
  }

  int32_t q31_input[] = {INT32_MIN, INT32_MAX, INT32_C(1) << 30, -(INT32_C(1) << 30), 17, -31, 63};
  int32_t q31_coefficients[] = {INT32_MIN, INT32_MAX, 1, -(INT32_C(1) << 29), INT32_C(1) << 28};
  int32_t q31_output[11] = {0};
  for (int64_t index = 0; index < 11; ++index) {
    int32_t actual =
        q31_full_filter_value(MEMREF_ARGS(q31_input, 7), MEMREF_ARGS(q31_coefficients, 5),
                              MEMREF_ARGS(q31_output, 11), index);
    int32_t expected = q31_reference(q31_input, 7, q31_coefficients, 5, index);
    if (actual != expected) {
      fprintf(stderr, "Q31 full output %lld: expected %d, got %d\n", (long long)index, expected,
              actual);
      failed = 1;
    }
  }

  int16_t short_input[] = {30000, -20000};
  int16_t long_coefficients[] = {16384, -8192, 4096, -2048, 1024};
  int16_t short_output[6] = {0};
  for (int64_t index = 0; index < 6; ++index) {
    int16_t actual =
        q15_full_filter_short_input(MEMREF_ARGS(short_input, 2), MEMREF_ARGS(long_coefficients, 5),
                                    MEMREF_ARGS(short_output, 6), index);
    int16_t expected = q15_reference(short_input, 2, long_coefficients, 5, index);
    if (actual != expected) {
      fprintf(stderr, "Q15 short full output %lld: expected %d, got %d\n", (long long)index,
              expected, actual);
      failed = 1;
    }
  }

  float f32_input[] = {-1.0f, 0x1.000002p+0f, 0.5f, -0.25f};
  float f32_coefficients[] = {1.0f, 0x1.fffffcp-1f, -0.5f};
  float f32_output[6] = {0.0f};
  for (int64_t output_index = 0; output_index < 6; ++output_index) {
    float expected = 0.0f;
    for (int64_t tap = 0; tap < 3; ++tap) {
      int64_t input_index = output_index + tap - 2;
      if (input_index >= 0 && input_index < 4)
        expected = fmaf(f32_input[input_index], f32_coefficients[tap], expected);
    }
    float actual =
        f32_full_filter_value(MEMREF_ARGS(f32_input, 4), MEMREF_ARGS(f32_coefficients, 3),
                              MEMREF_ARGS(f32_output, 6), output_index);
    if (f32_bits(actual) != f32_bits(expected)) {
      fprintf(stderr, "f32 full output %lld: expected 0x%08x, got 0x%08x\n",
              (long long)output_index, f32_bits(expected), f32_bits(actual));
      failed = 1;
    }
  }

  int16_t shared_coeff_init[] = {-32768, 16384, 8192, -4096, 2048};
  int16_t shared_coeff_original[5];
  memcpy(shared_coeff_original, shared_coeff_init, sizeof(shared_coeff_original));
  int16_t single_input[] = {24576};
  for (int64_t index = 0; index < 5; ++index) {
    memcpy(shared_coeff_init, shared_coeff_original, sizeof(shared_coeff_original));
    int16_t actual = q15_full_shared_coeff_init(MEMREF_ARGS(single_input, 1),
                                                MEMREF_ARGS(shared_coeff_init, 5), index);
    int16_t expected = q15_reference(single_input, 1, shared_coeff_original, 5, index);
    if (actual != expected) {
      fprintf(stderr, "shared coeff/init output %lld: expected %d, got %d\n", (long long)index,
              expected, actual);
      failed = 1;
    }
  }

  int16_t shared_input_init[] = {32767, -32768, 12345, -23456, 111, -222};
  int16_t shared_input_original[6];
  memcpy(shared_input_original, shared_input_init, sizeof(shared_input_original));
  int16_t unit_coefficient[] = {16384};
  for (int64_t index = 0; index < 6; ++index) {
    memcpy(shared_input_init, shared_input_original, sizeof(shared_input_original));
    int16_t actual = q15_full_shared_input_init(MEMREF_ARGS(shared_input_init, 6),
                                                MEMREF_ARGS(unit_coefficient, 1), index);
    int16_t expected = q15_reference(shared_input_original, 6, unit_coefficient, 1, index);
    if (actual != expected) {
      fprintf(stderr, "shared input/init output %lld: expected %d, got %d\n", (long long)index,
              expected, actual);
      failed = 1;
    }
  }

  return failed;
}
