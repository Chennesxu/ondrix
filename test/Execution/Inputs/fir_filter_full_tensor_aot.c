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
extern float f32_full_filter_value_off(float *, float *, int64_t, int64_t, int64_t, float *,
                                       float *, int64_t, int64_t, int64_t, float *, float *,
                                       int64_t, int64_t, int64_t, int64_t);
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

static float f32_reference(const float *input, int64_t input_length, const float *coefficients,
                           int64_t coefficient_length, int64_t output_index) {
  float accumulator = 0.0f;
  const int64_t left_padding = coefficient_length - 1;
  for (int64_t tap = 0; tap < coefficient_length; ++tap) {
    const int64_t input_index = output_index + tap - left_padding;
    if (input_index >= 0 && input_index < input_length)
      accumulator = fmaf(input[input_index], coefficients[tap], accumulator);
  }
  return accumulator;
}

// The off contract rounds each tap product to f32 before the accumulator adds
// it, so this reference must spell out a separate multiply and a separate add
// in the order the contract states. Using fmaf, or writing the tap as a single
// a * b + c expression, would express the fused contract instead. The RUN
// lines compile this file with -ffp-contract=off so the host compiler cannot
// fuse the two operations back together.
static float f32_off_reference(const float *input, int64_t input_length, const float *coefficients,
                               int64_t coefficient_length, int64_t output_index) {
  float accumulator = 0.0f;
  const int64_t left_padding = coefficient_length - 1;
  for (int64_t tap = 0; tap < coefficient_length; ++tap) {
    const int64_t input_index = output_index + tap - left_padding;
    if (input_index >= 0 && input_index < input_length) {
      float product = input[input_index] * coefficients[tap];
      accumulator = accumulator + product;
    }
  }
  return accumulator;
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
    float expected = f32_reference(f32_input, 4, f32_coefficients, 3, output_index);
    float actual =
        f32_full_filter_value(MEMREF_ARGS(f32_input, 4), MEMREF_ARGS(f32_coefficients, 3),
                              MEMREF_ARGS(f32_output, 6), output_index);
    if (f32_bits(actual) != f32_bits(expected)) {
      fprintf(stderr, "f32 full output %lld: expected 0x%08x, got 0x%08x\n",
              (long long)output_index, f32_bits(expected), f32_bits(actual));
      failed = 1;
    }

    float expected_off = f32_off_reference(f32_input, 4, f32_coefficients, 3, output_index);
    float actual_off =
        f32_full_filter_value_off(MEMREF_ARGS(f32_input, 4), MEMREF_ARGS(f32_coefficients, 3),
                                  MEMREF_ARGS(f32_output, 6), output_index);
    if (f32_bits(actual_off) != f32_bits(expected_off)) {
      fprintf(stderr, "f32 off full output %lld: expected 0x%08x, got 0x%08x\n",
              (long long)output_index, f32_bits(expected_off), f32_bits(actual_off));
      failed = 1;
    }
  }

  // The two taps of the middle window multiply out to -1.0 and 1 - 2^-46, so
  // the separate multiply rounds the second product to 1.0 and cancels to
  // zero while the fused update keeps the residual. This corpus is what makes
  // the off gate observe the contract rather than merely repeat the fused one.
  float contract_input[] = {-1.0f, 0x1.000002p+0f};
  float contract_coefficients[] = {1.0f, 0x1.fffffcp-1f};
  float contract_output[3] = {0.0f};
  for (int64_t output_index = 0; output_index < 3; ++output_index) {
    float expected = f32_off_reference(contract_input, 2, contract_coefficients, 2, output_index);
    float actual = f32_full_filter_value_off(MEMREF_ARGS(contract_input, 2),
                                             MEMREF_ARGS(contract_coefficients, 2),
                                             MEMREF_ARGS(contract_output, 3), output_index);
    if (f32_bits(actual) != f32_bits(expected)) {
      fprintf(stderr, "f32 off contract output %lld: expected 0x%08x, got 0x%08x\n",
              (long long)output_index, f32_bits(expected), f32_bits(actual));
      failed = 1;
    }
  }
  float cancelled = f32_off_reference(contract_input, 2, contract_coefficients, 2, 1);
  if (f32_bits(cancelled) != UINT32_C(0)) {
    fprintf(stderr, "f32 off contract reference lost the cancellation: 0x%08x\n",
            f32_bits(cancelled));
    failed = 1;
  }

  float exceptional_input[] = {1.0f};
  float exceptional_coefficients[] = {NAN, INFINITY, 1.0f};
  float exceptional_output[3] = {0.0f};
  for (int64_t output_index = 0; output_index < 3; ++output_index) {
    float expected = f32_reference(exceptional_input, 1, exceptional_coefficients, 3, output_index);
    float actual = f32_full_filter_value(MEMREF_ARGS(exceptional_input, 1),
                                         MEMREF_ARGS(exceptional_coefficients, 3),
                                         MEMREF_ARGS(exceptional_output, 3), output_index);
    int matches = isnan(expected) ? isnan(actual) : f32_bits(actual) == f32_bits(expected);
    if (!matches) {
      fprintf(stderr, "exceptional f32 full output %lld: expected 0x%08x, got 0x%08x\n",
              (long long)output_index, f32_bits(expected), f32_bits(actual));
      failed = 1;
    }

    float expected_off =
        f32_off_reference(exceptional_input, 1, exceptional_coefficients, 3, output_index);
    float actual_off = f32_full_filter_value_off(MEMREF_ARGS(exceptional_input, 1),
                                                 MEMREF_ARGS(exceptional_coefficients, 3),
                                                 MEMREF_ARGS(exceptional_output, 3), output_index);
    int matches_off =
        isnan(expected_off) ? isnan(actual_off) : f32_bits(actual_off) == f32_bits(expected_off);
    if (!matches_off) {
      fprintf(stderr, "exceptional f32 off full output %lld: expected 0x%08x, got 0x%08x\n",
              (long long)output_index, f32_bits(expected_off), f32_bits(actual_off));
      failed = 1;
    }
  }

  float signed_zero_input[] = {-0.0f};
  float signed_zero_coefficients[] = {1.0f};
  float signed_zero_output[] = {0.0f};
  float signed_zero_expected = f32_reference(signed_zero_input, 1, signed_zero_coefficients, 1, 0);
  float signed_zero_actual = f32_full_filter_value(MEMREF_ARGS(signed_zero_input, 1),
                                                   MEMREF_ARGS(signed_zero_coefficients, 1),
                                                   MEMREF_ARGS(signed_zero_output, 1), 0);
  if (f32_bits(signed_zero_actual) != f32_bits(signed_zero_expected)) {
    fprintf(stderr, "signed-zero f32 full output: expected 0x%08x, got 0x%08x\n",
            f32_bits(signed_zero_expected), f32_bits(signed_zero_actual));
    failed = 1;
  }

  float signed_zero_off_expected =
      f32_off_reference(signed_zero_input, 1, signed_zero_coefficients, 1, 0);
  float signed_zero_off_actual = f32_full_filter_value_off(MEMREF_ARGS(signed_zero_input, 1),
                                                           MEMREF_ARGS(signed_zero_coefficients, 1),
                                                           MEMREF_ARGS(signed_zero_output, 1), 0);
  if (f32_bits(signed_zero_off_actual) != f32_bits(signed_zero_off_expected)) {
    fprintf(stderr, "signed-zero f32 off full output: expected 0x%08x, got 0x%08x\n",
            f32_bits(signed_zero_off_expected), f32_bits(signed_zero_off_actual));
    failed = 1;
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
