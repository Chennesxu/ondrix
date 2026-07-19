#include <limits.h>
#include <stdint.h>
#include <stdio.h>

typedef void (*fir_kernel_t)(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                             int64_t, int64_t, int64_t, int16_t *, int16_t *, int64_t, int64_t,
                             int64_t);

extern void q15_full_output_vector(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                   int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                   int64_t, int64_t, int64_t);
extern void q15_full_output_scalar(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                   int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                   int64_t, int64_t, int64_t);
extern void q15_full_boundary_scalar(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *,
                                     int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                                     int64_t, int64_t, int64_t);

static int64_t saturating_update(int64_t accumulator, int16_t input, int16_t coefficient) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  const __int128 update = (__int128)accumulator + (__int128)input * coefficient;
  if (update < minimum)
    return (int64_t)minimum;
  if (update > maximum)
    return (int64_t)maximum;
  return (int64_t)update;
}

static uint16_t export_floor_wrap(int64_t accumulator) {
  const int64_t divisor = INT64_C(1) << 15;
  int64_t quotient = accumulator / divisor;
  if (accumulator < 0 && accumulator % divisor != 0)
    --quotient;
  return (uint16_t)quotient;
}

static uint16_t reference_sample(const int16_t *input, int64_t input_offset, int64_t input_stride,
                                 const int16_t *coefficients, int64_t coefficient_offset,
                                 int64_t coefficient_stride, int64_t tap_count) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < tap_count; ++tap) {
    accumulator = saturating_update(accumulator, input[input_offset + tap * input_stride],
                                    coefficients[coefficient_offset + tap * coefficient_stride]);
  }
  return export_floor_wrap(accumulator);
}

static void run_kernel(fir_kernel_t kernel, int16_t *input, int64_t input_offset,
                       int64_t input_length, int64_t input_stride, int16_t *coefficients,
                       int64_t coefficient_offset, int64_t coefficient_length,
                       int64_t coefficient_stride, int16_t *output, int64_t output_offset,
                       int64_t output_length, int64_t output_stride) {
  kernel(input, input, input_offset, input_length, input_stride, coefficients, coefficients,
         coefficient_offset, coefficient_length, coefficient_stride, output, output, output_offset,
         output_length, output_stride);
}

static int check_case(const char *name, fir_kernel_t kernel, int16_t *input, int64_t input_offset,
                      int64_t input_length, int64_t input_stride, int16_t *coefficients,
                      int64_t coefficient_offset, int64_t coefficient_length,
                      int64_t coefficient_stride, int16_t *output, int64_t output_offset,
                      int64_t output_stride) {
  const int64_t output_length = input_length - coefficient_length + 1;
  run_kernel(kernel, input, input_offset, input_length, input_stride, coefficients,
             coefficient_offset, coefficient_length, coefficient_stride, output, output_offset,
             output_length, output_stride);

  for (int64_t index = 0; index < output_length; ++index) {
    uint16_t expected =
        reference_sample(input, input_offset + index * input_stride, input_stride, coefficients,
                         coefficient_offset, coefficient_stride, coefficient_length);
    uint16_t actual = (uint16_t)output[output_offset + index * output_stride];
    if (actual == expected)
      continue;
    fprintf(stderr, "%s output %lld: expected 0x%04x, got 0x%04x\n", name, (long long)index,
            expected, actual);
    return 1;
  }
  return 0;
}

static int check_full_boundary(int16_t *input, int64_t input_offset, int64_t input_length,
                               int64_t input_stride, int16_t *coefficients,
                               int64_t coefficient_offset, int64_t coefficient_length,
                               int64_t coefficient_stride, int16_t *output, int64_t output_offset,
                               int64_t output_stride) {
  const int64_t output_length = input_length + coefficient_length - 1;
  run_kernel(q15_full_boundary_scalar, input, input_offset, input_length, input_stride,
             coefficients, coefficient_offset, coefficient_length, coefficient_stride, output,
             output_offset, output_length, output_stride);

  for (int64_t output_index = 0; output_index < output_length; ++output_index) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < coefficient_length; ++tap) {
      int64_t padded_index = output_index + tap;
      if (padded_index < coefficient_length - 1)
        continue;
      int64_t input_index = padded_index - (coefficient_length - 1);
      if (input_index >= input_length)
        continue;
      accumulator = saturating_update(accumulator, input[input_offset + input_index * input_stride],
                                      coefficients[coefficient_offset + tap * coefficient_stride]);
    }
    uint16_t expected = export_floor_wrap(accumulator);
    uint16_t actual = (uint16_t)output[output_offset + output_index * output_stride];
    if (actual == expected)
      continue;
    fprintf(stderr, "full boundary output %lld: expected 0x%04x, got 0x%04x\n",
            (long long)output_index, expected, actual);
    return 1;
  }
  return 0;
}

int main(void) {
  int16_t input[520];
  int16_t coefficients[520];
  int16_t vector_output[80];
  int16_t scalar_output[1100];
  uint32_t state = UINT32_C(0x9e3779b9);
  for (unsigned index = 0; index < 520; ++index) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    input[index] = (int16_t)(state >> 16);
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    coefficients[index] = (int16_t)(state >> 16);
  }
  for (unsigned index = 0; index < 80; ++index) {
    vector_output[index] = (int16_t)0x5555;
    scalar_output[index] = (int16_t)0x5555;
  }

  int failed = 0;
  failed |= check_case("unit-stride vector", q15_full_output_vector, input, 1, 25, 1, coefficients,
                       2, 9, 1, vector_output, 3, 1);
  failed |= check_case("dynamic-stride scalar", q15_full_output_scalar, input, 1, 17, 2,
                       coefficients, 2, 5, 3, scalar_output, 2, 2);
  failed |= check_full_boundary(input, 1, 5, 2, coefficients, 2, 4, 3, scalar_output, 3, 2);

  for (unsigned index = 0; index < 520; ++index)
    input[index] = INT16_MIN;
  for (unsigned index = 0; index < 520; ++index)
    coefficients[index] = INT16_MIN;
  failed |= check_case("overflow vector", q15_full_output_vector, input, 0, 520, 1, coefficients, 0,
                       512, 1, vector_output, 0, 1);

  // Lane 0 of the last chunk reaches positive saturation and lane 1 applies a
  // negative product. Reassociating or reordering the chunk changes the result.
  coefficients[513] = INT16_MAX;
  coefficients[514] = 0;
  coefficients[515] = 0;
  failed |= check_case("ordered vector", q15_full_output_vector, input, 0, 520, 1, coefficients, 0,
                       516, 1, vector_output, 0, 1);
  failed |= check_full_boundary(input, 0, 520, 1, coefficients, 0, 516, 1, scalar_output, 0, 1);
  return failed;
}
