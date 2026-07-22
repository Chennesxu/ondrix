#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef float (*DotKernel)(float *, float *, int64_t, int64_t, int64_t, float *, float *, int64_t,
                           int64_t, int64_t);

extern float f32_dot_off(float *, float *, int64_t, int64_t, int64_t, float *, float *, int64_t,
                         int64_t, int64_t);
extern float f32_dot_fma(float *, float *, int64_t, int64_t, int64_t, float *, float *, int64_t,
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

static float reference_off(const float *lhs, const float *rhs, int64_t count) {
  float accumulator = 0.0f;
  for (int64_t i = 0; i < count; ++i) {
    volatile float product = lhs[i] * rhs[i];
    accumulator = accumulator + product;
  }
  return accumulator;
}

static float reference_fma(const float *lhs, const float *rhs, int64_t count) {
  float accumulator = 0.0f;
  for (int64_t i = 0; i < count; ++i)
    accumulator = fmaf(lhs[i], rhs[i], accumulator);
  return accumulator;
}

static int run_case(const char *name, DotKernel kernel,
                    float (*reference)(const float *, const float *, int64_t), float *lhs,
                    float *rhs, int64_t count) {
  float actual = kernel(lhs, lhs, 0, count, 1, rhs, rhs, 0, count, 1);
  float expected = reference(lhs, rhs, count);
  if ((isnan(actual) && isnan(expected)) || bits(actual) == bits(expected))
    return 0;
  fprintf(stderr, "%s: got 0x%08x, expected 0x%08x\n", name, bits(actual), bits(expected));
  return 1;
}

int main(void) {
  float lhs[] = {1.0f, -3.25f, 0.0f, -0.0f, 0x1.000002p0f, 0x1.fffffep20f};
  float rhs[] = {2.0f, 0.5f, -1.0f, 1.0f, 0x1.fffffep-1f, -0x1.000002p-20f};
  int64_t lengths[] = {0, 1, 2, 4, 6};

  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t length = lengths[i];
    if (run_case("off", f32_dot_off, reference_off, lhs, rhs, length) ||
        run_case("fma", f32_dot_fma, reference_fma, lhs, rhs, length))
      return 1;
  }

  lhs[0] = from_bits(UINT32_C(0x7fc12345));
  rhs[0] = 1.0f;
  if (run_case("off-nan", f32_dot_off, reference_off, lhs, rhs, 1) ||
      run_case("fma-nan", f32_dot_fma, reference_fma, lhs, rhs, 1))
    return 1;
  return 0;
}
