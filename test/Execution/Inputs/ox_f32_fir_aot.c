#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef float (*FirKernel)(float *, float *, int64_t, int64_t, int64_t, float *, float *, int64_t,
                           int64_t, int64_t);

extern float f32_fir_off(float *, float *, int64_t, int64_t, int64_t, float *, float *, int64_t,
                         int64_t, int64_t);
extern float f32_fir_fma(float *, float *, int64_t, int64_t, int64_t, float *, float *, int64_t,
                         int64_t, int64_t);

static uint32_t bits(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static float from_bits(uint32_t value) {
  float result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static float reference_off(const float *window, const float *coefficients, int64_t count) {
  float accumulator = 0.0f;
  for (int64_t i = 0; i < count; ++i) {
    volatile float product = window[i] * coefficients[i];
    accumulator = accumulator + product;
  }
  return accumulator;
}

static float reference_fma(const float *window, const float *coefficients, int64_t count) {
  float accumulator = 0.0f;
  for (int64_t i = 0; i < count; ++i)
    accumulator = fmaf(window[i], coefficients[i], accumulator);
  return accumulator;
}

static int run_case(const char *name, FirKernel kernel,
                    float (*reference)(const float *, const float *, int64_t), float *window,
                    float *coefficients, int64_t count) {
  float actual = kernel(window, window, 0, count, 1, coefficients, coefficients, 0, count, 1);
  float expected = reference(window, coefficients, count);
  if ((isnan(actual) && isnan(expected)) || bits(actual) == bits(expected))
    return 0;
  fprintf(stderr, "%s: got 0x%08x, expected 0x%08x\n", name, bits(actual), bits(expected));
  return 1;
}

int main(void) {
  float window[] = {1.0f, -3.25f, 0.0f, -0.0f, 0x1.000002p0f, 0x1.fffffep20f};
  float coefficients[] = {2.0f, 0.5f, -1.0f, 1.0f, 0x1.fffffep-1f, -0x1.000002p-20f};
  int64_t lengths[] = {0, 1, 2, 4, 6};

  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t length = lengths[i];
    if (run_case("off", f32_fir_off, reference_off, window, coefficients, length) ||
        run_case("fma", f32_fir_fma, reference_fma, window, coefficients, length))
      return 1;
  }

  window[0] = from_bits(UINT32_C(0x7fc12345));
  coefficients[0] = 1.0f;
  if (run_case("off-nan", f32_fir_off, reference_off, window, coefficients, 1) ||
      run_case("fma-nan", f32_fir_fma, reference_fma, window, coefficients, 1))
    return 1;
  return 0;
}
