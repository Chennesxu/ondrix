#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void (*q31_kernel_t)(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
                             int64_t, int64_t, int64_t, int32_t *, int32_t *, int64_t, int64_t,
                             int64_t);
typedef void (*f32_kernel_t)(float *, float *, int64_t, int64_t, int64_t, float *, float *, int64_t,
                             int64_t, int64_t, float *, float *, int64_t, int64_t, int64_t);

extern void q31_full_output_vector(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                   int32_t *, int64_t, int64_t, int64_t, int32_t *, int32_t *,
                                   int64_t, int64_t, int64_t);
extern void q31_full_output_raw_vector(int32_t *, int32_t *, int64_t, int64_t, int64_t, int32_t *,
                                       int32_t *, int64_t, int64_t, int64_t, int64_t *, int64_t *,
                                       int64_t, int64_t, int64_t);
extern void f32_full_output_scalar(float *, float *, int64_t, int64_t, int64_t, float *, float *,
                                   int64_t, int64_t, int64_t, float *, float *, int64_t, int64_t,
                                   int64_t);
extern void f32_full_output_scalar_off(float *, float *, int64_t, int64_t, int64_t, float *,
                                       float *, int64_t, int64_t, int64_t, float *, float *,
                                       int64_t, int64_t, int64_t);

static int64_t q31_update(int64_t accumulator, int32_t input, int32_t coefficient) {
  __int128 update = (__int128)accumulator + (__int128)input * coefficient;
  if (update < INT64_MIN)
    return INT64_MIN;
  if (update > INT64_MAX)
    return INT64_MAX;
  return (int64_t)update;
}

static __int128 floor_divide(__int128 value, unsigned shift) {
  __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  if (value < 0 && value % divisor != 0)
    --quotient;
  return quotient;
}

static int32_t q31_export_nearest_even(int64_t accumulator) {
  __int128 quotient = floor_divide(accumulator, 31);
  __int128 divisor = (__int128)1 << 31;
  __int128 remainder = (__int128)accumulator - quotient * divisor;
  __int128 half = divisor / 2;
  if (remainder > half || (remainder == half && (quotient & 1) != 0))
    ++quotient;
  if (quotient < INT32_MIN)
    return INT32_MIN;
  if (quotient > INT32_MAX)
    return INT32_MAX;
  return (int32_t)quotient;
}

static int32_t q31_reference_sample(const int32_t *input, int64_t input_offset,
                                    int64_t input_stride, const int32_t *coefficients,
                                    int64_t coefficient_offset, int64_t coefficient_stride,
                                    int64_t tap_count) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < tap_count; ++tap) {
    accumulator = q31_update(accumulator, input[input_offset + tap * input_stride],
                             coefficients[coefficient_offset + tap * coefficient_stride]);
  }
  return q31_export_nearest_even(accumulator);
}

static int64_t q31_reference_raw(const int32_t *input, int64_t input_offset, int64_t input_stride,
                                 const int32_t *coefficients, int64_t coefficient_offset,
                                 int64_t coefficient_stride, int64_t tap_count) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < tap_count; ++tap) {
    accumulator = q31_update(accumulator, input[input_offset + tap * input_stride],
                             coefficients[coefficient_offset + tap * coefficient_stride]);
  }
  return accumulator;
}

static int32_t signed_i32_bits(uint32_t bits) {
  int32_t value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float f32_reference_sample(const float *input, int64_t input_offset, int64_t input_stride,
                                  const float *coefficients, int64_t coefficient_offset,
                                  int64_t coefficient_stride, int64_t tap_count) {
  float accumulator = 0.0f;
  for (int64_t tap = 0; tap < tap_count; ++tap) {
    accumulator = fmaf(input[input_offset + tap * input_stride],
                       coefficients[coefficient_offset + tap * coefficient_stride], accumulator);
  }
  return accumulator;
}

// The off contract rounds each tap product to f32 before the accumulator adds
// it, so this reference must spell out a separate multiply and a separate add
// in the order the contract states. Using fmaf, or writing the tap as a single
// a * b + c expression, would express the fused contract instead. The RUN line
// compiles this file with -ffp-contract=off so the host compiler cannot fuse
// the two operations back together.
static float f32_off_reference_sample(const float *input, int64_t input_offset,
                                      int64_t input_stride, const float *coefficients,
                                      int64_t coefficient_offset, int64_t coefficient_stride,
                                      int64_t tap_count) {
  float accumulator = 0.0f;
  for (int64_t tap = 0; tap < tap_count; ++tap) {
    float product = input[input_offset + tap * input_stride] *
                    coefficients[coefficient_offset + tap * coefficient_stride];
    accumulator = accumulator + product;
  }
  return accumulator;
}

static int check_q31(int32_t *input, int64_t input_offset, int64_t input_length,
                     int32_t *coefficients, int64_t coefficient_offset, int64_t coefficient_length,
                     int32_t *output, int64_t output_offset) {
  int64_t output_length = input_length - coefficient_length + 1;
  q31_full_output_vector(input, input, input_offset, input_length, 1, coefficients, coefficients,
                         coefficient_offset, coefficient_length, 1, output, output, output_offset,
                         output_length, 1);
  for (int64_t index = 0; index < output_length; ++index) {
    int32_t expected = q31_reference_sample(input, input_offset + index, 1, coefficients,
                                            coefficient_offset, 1, coefficient_length);
    int32_t actual = output[output_offset + index];
    if (actual == expected)
      continue;
    fprintf(stderr, "Q31 output %lld: expected %d, got %d\n", (long long)index, expected, actual);
    return 1;
  }
  return 0;
}

static int check_q31_raw(int32_t *input, int64_t input_length, int32_t *coefficients,
                         int64_t coefficient_length, int64_t *output) {
  int64_t output_length = input_length - coefficient_length + 1;
  q31_full_output_raw_vector(input, input, 0, input_length, 1, coefficients, coefficients, 0,
                             coefficient_length, 1, output, output, 0, output_length, 1);
  for (int64_t index = 0; index < output_length; ++index) {
    int64_t expected = q31_reference_raw(input, index, 1, coefficients, 0, 1, coefficient_length);
    if (output[index] == expected)
      continue;
    fprintf(stderr, "Q31 raw output %lld: expected %lld, got %lld\n", (long long)index,
            (long long)expected, (long long)output[index]);
    return 1;
  }
  return 0;
}

static int check_f32(float *input, int64_t input_offset, int64_t input_length, int64_t input_stride,
                     float *coefficients, int64_t coefficient_offset, int64_t coefficient_length,
                     int64_t coefficient_stride, float *output, int64_t output_offset,
                     int64_t output_stride) {
  int64_t output_length = input_length - coefficient_length + 1;
  f32_full_output_scalar(input, input, input_offset, input_length, input_stride, coefficients,
                         coefficients, coefficient_offset, coefficient_length, coefficient_stride,
                         output, output, output_offset, output_length, output_stride);
  for (int64_t index = 0; index < output_length; ++index) {
    float expected =
        f32_reference_sample(input, input_offset + index * input_stride, input_stride, coefficients,
                             coefficient_offset, coefficient_stride, coefficient_length);
    float actual = output[output_offset + index * output_stride];
    if (f32_bits(actual) == f32_bits(expected))
      continue;
    fprintf(stderr, "f32 output %lld: expected 0x%08x, got 0x%08x\n", (long long)index,
            f32_bits(expected), f32_bits(actual));
    return 1;
  }
  return 0;
}

static int check_f32_off(float *input, int64_t input_offset, int64_t input_length,
                         int64_t input_stride, float *coefficients, int64_t coefficient_offset,
                         int64_t coefficient_length, int64_t coefficient_stride, float *output,
                         int64_t output_offset, int64_t output_stride) {
  int64_t output_length = input_length - coefficient_length + 1;
  f32_full_output_scalar_off(input, input, input_offset, input_length, input_stride, coefficients,
                             coefficients, coefficient_offset, coefficient_length,
                             coefficient_stride, output, output, output_offset, output_length,
                             output_stride);
  for (int64_t index = 0; index < output_length; ++index) {
    float expected = f32_off_reference_sample(input, input_offset + index * input_stride,
                                              input_stride, coefficients, coefficient_offset,
                                              coefficient_stride, coefficient_length);
    float actual = output[output_offset + index * output_stride];
    if (f32_bits(actual) == f32_bits(expected))
      continue;
    fprintf(stderr, "f32 off output %lld: expected 0x%08x, got 0x%08x\n", (long long)index,
            f32_bits(expected), f32_bits(actual));
    return 1;
  }
  return 0;
}

int main(void) {
  int32_t q31_input[32];
  int32_t q31_coefficients[16];
  int32_t q31_output[32] = {0};
  uint32_t state = UINT32_C(0x243f6a88);
  for (unsigned index = 0; index < 32; ++index) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    q31_input[index] = signed_i32_bits(state);
  }
  for (unsigned index = 0; index < 16; ++index) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    q31_coefficients[index] = signed_i32_bits(state);
  }

  int failed = check_q31(q31_input, 1, 25, q31_coefficients, 2, 9, q31_output, 3);
  for (unsigned index = 0; index < 32; ++index)
    q31_input[index] = INT32_MIN;
  q31_coefficients[0] = INT32_MIN;
  q31_coefficients[1] = INT32_MIN;
  q31_coefficients[2] = INT32_MAX;
  q31_coefficients[3] = 0;
  failed |= check_q31(q31_input, 0, 8, q31_coefficients, 0, 4, q31_output, 0);
  int64_t q31_raw_output[8] = {0};
  failed |= check_q31_raw(q31_input, 8, q31_coefficients, 4, q31_raw_output);

  int32_t tie_input[3] = {INT32_C(1) << 30, INT32_C(1) << 30, INT32_C(1) << 30};
  int32_t negative_tie_input[3] = {-(INT32_C(1) << 30), -(INT32_C(1) << 30), -(INT32_C(1) << 30)};
  int32_t tie_coefficients[3] = {1, 1, 1};
  failed |= check_q31(tie_input, 0, 1, tie_coefficients, 0, 1, q31_output, 0);
  failed |= check_q31(tie_input, 0, 3, tie_coefficients, 0, 3, q31_output, 0);
  failed |= check_q31(negative_tie_input, 0, 1, tie_coefficients, 0, 1, q31_output, 0);
  failed |= check_q31(negative_tie_input, 0, 3, tie_coefficients, 0, 3, q31_output, 0);

  float f32_input[64];
  float f32_coefficients[24];
  float f32_output[64] = {0.0f};
  for (unsigned index = 0; index < 64; ++index)
    f32_input[index] = (float)((int)(index % 13) - 6) * 0.15625f;
  for (unsigned index = 0; index < 24; ++index)
    f32_coefficients[index] = (float)((int)(index % 7) - 3) * 0.09375f;
  f32_input[1] = -0.0f;
  f32_coefficients[2] = 1.0f;
  failed |= check_f32(f32_input, 1, 17, 2, f32_coefficients, 2, 5, 3, f32_output, 3, 2);
  failed |= check_f32_off(f32_input, 1, 17, 2, f32_coefficients, 2, 5, 3, f32_output, 3, 2);

  // A separate multiply rounds the second product to 1.0 and cancellation
  // produces zero. The required fused update preserves the -2^-46 residual.
  // The two contract modes therefore disagree on this corpus, which is what
  // makes the off gate observe the contract rather than repeat the fused one.
  f32_input[0] = -1.0f;
  f32_input[1] = 0x1.000002p+0f;
  f32_coefficients[0] = 1.0f;
  f32_coefficients[1] = 0x1.fffffcp-1f;
  failed |= check_f32(f32_input, 0, 2, 1, f32_coefficients, 0, 2, 1, f32_output, 0, 1);
  failed |= check_f32_off(f32_input, 0, 2, 1, f32_coefficients, 0, 2, 1, f32_output, 0, 1);
  return failed;
}
