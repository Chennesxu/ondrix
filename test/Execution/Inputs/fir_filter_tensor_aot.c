#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MEMREF_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)

extern int16_t q15_fir_filter_value(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                    int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                    int64_t, int64_t, int64_t, int64_t);
extern int32_t q31_fir_filter_value(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                    int32_t *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
                                    int64_t, int64_t, int64_t, int64_t);
extern int16_t q15_fir_filter_shared_coeff_init(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                                int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                                int64_t);
extern float f32_fir_filter_value(float *, float *, int64_t, int64_t, int64_t, float *, float *,
                                  int64_t, int64_t, int64_t, float *, float *, int64_t, int64_t,
                                  int64_t, int64_t);
extern int16_t q15_proven_fir_filter_value(int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                           int16_t *, int16_t *, int64_t, int64_t, int64_t,
                                           int64_t);
extern int32_t q31_proven_fir_filter_value(int32_t *, int32_t *, int64_t, int64_t, int64_t,
                                           int32_t *, int32_t *, int64_t, int64_t, int64_t,
                                           int64_t);

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
  __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  __int128 remainder = value % divisor;
  if (remainder < 0) {
    remainder += divisor;
    --quotient;
  }
  __int128 half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1) != 0))
    ++quotient;
  return quotient;
}

static int16_t q15_reference(const int16_t *input, const int16_t *coeffs, int64_t output_index,
                             int64_t tap_count) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < tap_count; ++tap)
    accumulator =
        clamp_i40((__int128)accumulator + (__int128)input[output_index + tap] * coeffs[tap]);
  __int128 result = round_nearest_even(accumulator, 15);
  if (result < INT16_MIN)
    return INT16_MIN;
  if (result > INT16_MAX)
    return INT16_MAX;
  return (int16_t)result;
}

static int32_t q31_reference(const int32_t *input, const int32_t *coeffs, int64_t output_index,
                             int64_t tap_count) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < tap_count; ++tap)
    accumulator =
        clamp_i64((__int128)accumulator + (__int128)input[output_index + tap] * coeffs[tap]);
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

  int16_t q15_input[] = {32767, -32768, 4096, -8192, 16384, 1, -2, 3, -5, 7};
  int16_t q15_coeffs[] = {-32768, 16384, 8192, -4096, 2048};
  int16_t q15_init[6] = {11, 12, 13, 14, 15, 16};
  for (int64_t index = 0; index < 6; ++index) {
    int16_t actual = q15_fir_filter_value(MEMREF_ARGS(q15_input, 10), MEMREF_ARGS(q15_coeffs, 5),
                                          MEMREF_ARGS(q15_init, 6), index);
    int16_t expected = q15_reference(q15_input, q15_coeffs, index, 5);
    if (actual != expected) {
      fprintf(stderr, "Q15 output %lld: expected %d, got %d\n", (long long)index, expected, actual);
      failed = 1;
    }
  }

  int16_t shared_coeffs_and_init[] = {-32768, 16384, 8192, -4096, 2048};
  int16_t shared_original[5];
  memcpy(shared_original, shared_coeffs_and_init, sizeof(shared_original));
  for (int64_t index = 0; index < 5; ++index) {
    int16_t actual = q15_fir_filter_shared_coeff_init(
        MEMREF_ARGS(q15_input, 9), MEMREF_ARGS(shared_coeffs_and_init, 5), index);
    int16_t expected = q15_reference(q15_input, shared_original, index, 5);
    if (actual != expected) {
      fprintf(stderr, "Q15 shared operand %lld: expected %d, got %d\n", (long long)index, expected,
              actual);
      failed = 1;
    }
    memcpy(shared_coeffs_and_init, shared_original, sizeof(shared_original));
  }

  int32_t q31_input[] = {INT32_MIN, INT32_MAX, INT32_C(1) << 30, -(INT32_C(1) << 30), 17, -31, 63};
  int32_t q31_coeffs[] = {INT32_MIN, INT32_MAX, 1, -(INT32_C(1) << 29), INT32_C(1) << 28};
  int32_t q31_init[3] = {101, 102, 103};
  for (int64_t index = 0; index < 3; ++index) {
    int32_t actual = q31_fir_filter_value(MEMREF_ARGS(q31_input, 7), MEMREF_ARGS(q31_coeffs, 5),
                                          MEMREF_ARGS(q31_init, 3), index);
    int32_t expected = q31_reference(q31_input, q31_coeffs, index, 5);
    if (actual != expected) {
      fprintf(stderr, "Q31 output %lld: expected %d, got %d\n", (long long)index, expected, actual);
      failed = 1;
    }
  }

  float f32_input[] = {-1.0f, 0x1.000002p+0f, 0.5f};
  float f32_coeffs[] = {1.0f, 0x1.fffffcp-1f};
  float f32_init[2] = {7.0f, 8.0f};
  for (int64_t index = 0; index < 2; ++index) {
    float expected = 0.0f;
    for (int64_t tap = 0; tap < 2; ++tap)
      expected = fmaf(f32_input[index + tap], f32_coeffs[tap], expected);
    float actual = f32_fir_filter_value(MEMREF_ARGS(f32_input, 3), MEMREF_ARGS(f32_coeffs, 2),
                                        MEMREF_ARGS(f32_init, 2), index);
    if (f32_bits(actual) != f32_bits(expected)) {
      fprintf(stderr, "f32 output %lld: expected 0x%08x, got 0x%08x\n", (long long)index,
              f32_bits(expected), f32_bits(actual));
      failed = 1;
    }
  }

  const int16_t q15_proven_coefficients[] = {-4096, 2048, -1024, 512, -256, 128, -64, 32};
  int16_t q15_proven_input[] = {32767, -32768, 24576, -16384, 8192, -4096,
                                2048,  -1024,  512,   -256,   128,  -64};
  int16_t q15_proven_init[5] = {0};
  for (int64_t index = 0; index < 5; ++index) {
    int16_t actual = q15_proven_fir_filter_value(MEMREF_ARGS(q15_proven_input, 12),
                                                 MEMREF_ARGS(q15_proven_init, 5), index);
    int16_t expected = q15_reference(q15_proven_input, q15_proven_coefficients, index, 8);
    if (actual != expected) {
      fprintf(stderr, "Q15 proven output %lld: expected %d, got %d\n", (long long)index, expected,
              actual);
      failed = 1;
    }
  }

  const int32_t q31_proven_coefficients[] = {INT32_C(1) << 26, -(INT32_C(1) << 25),
                                             INT32_C(1) << 24, -(INT32_C(1) << 23)};
  int32_t q31_proven_input[] = {INT32_MIN, INT32_MAX, INT32_C(1) << 30, -(INT32_C(1) << 29), 17,
                                -31,       63};
  int32_t q31_proven_init[4] = {0};
  for (int64_t index = 0; index < 4; ++index) {
    int32_t actual = q31_proven_fir_filter_value(MEMREF_ARGS(q31_proven_input, 7),
                                                 MEMREF_ARGS(q31_proven_init, 4), index);
    int32_t expected = q31_reference(q31_proven_input, q31_proven_coefficients, index, 4);
    if (actual != expected) {
      fprintf(stderr, "Q31 proven output %lld: expected %d, got %d\n", (long long)index, expected,
              actual);
      failed = 1;
    }
  }

  return failed;
}
