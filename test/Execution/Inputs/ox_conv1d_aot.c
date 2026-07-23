#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DECLARE_MEMREF(NAME, TYPE)                                                                 \
  typedef struct {                                                                                 \
    TYPE *allocated;                                                                               \
    TYPE *aligned;                                                                                 \
    int64_t offset;                                                                                \
    int64_t sizes[1];                                                                              \
    int64_t strides[1];                                                                            \
  } NAME

DECLARE_MEMREF(MemRefI16, int16_t);
DECLARE_MEMREF(MemRefF32, float);

extern void _mlir_ciface_q15_convolution(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_f32_correlation(MemRefF32 *, MemRefF32 *, MemRefF32 *);

static int64_t clampI40(__int128 value) {
  const __int128 minimum = -((__int128)1 << 39);
  const __int128 maximum = ((__int128)1 << 39) - 1;
  if (value < minimum)
    return (int64_t)minimum;
  if (value > maximum)
    return (int64_t)maximum;
  return (int64_t)value;
}

static int16_t exportQ15(int64_t accumulator) {
  const __int128 divisor = (__int128)1 << 15;
  __int128 quotient = (__int128)accumulator / divisor;
  __int128 remainder = (__int128)accumulator % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  const __int128 half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient < INT16_MIN)
    return INT16_MIN;
  if (quotient > INT16_MAX)
    return INT16_MAX;
  return (int16_t)quotient;
}

static int16_t q15Convolution(const int16_t input[6], const int16_t kernel[3], unsigned output) {
  int64_t accumulator = 0;
  for (unsigned k = 0; k < 3; ++k)
    accumulator = clampI40((__int128)accumulator + (__int128)input[output + k] * kernel[2 - k]);
  return exportQ15(accumulator);
}

static float f32Correlation(const float input[6], const float kernel[3], unsigned output) {
  float accumulator = 0.0f;
  for (unsigned k = 0; k < 3; ++k)
    accumulator = fmaf(input[output + k], kernel[k], accumulator);
  return accumulator;
}

static uint32_t f32Bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int main(void) {
  int16_t q15Input[] = {32767, -32768, 16384, -8192, 4096, -2048};
  int16_t q15Kernel[] = {16384, -8192, 4096};
  MemRefI16 q15InputRef = {q15Input, q15Input, 0, {6}, {1}};
  MemRefI16 q15KernelRef = {q15Kernel, q15Kernel, 0, {3}, {1}};
  MemRefI16 q15Output;
  _mlir_ciface_q15_convolution(&q15Output, &q15InputRef, &q15KernelRef);
  if (q15Output.sizes[0] != 4)
    return 1;
  for (unsigned output = 0; output < 4; ++output) {
    int16_t actual = q15Output.aligned[q15Output.offset + output * q15Output.strides[0]];
    int16_t expected = q15Convolution(q15Input, q15Kernel, output);
    if (actual != expected) {
      fprintf(stderr, "Q15 convolution output %u: %d/%d\n", output, actual, expected);
      free(q15Output.allocated);
      return 1;
    }
  }
  free(q15Output.allocated);

  float f32Input[] = {1.25f, -2.0f, 3.5f, -4.25f, 5.0f, -6.5f};
  float f32Kernel[] = {0.5f, -1.25f, 2.0f};
  MemRefF32 f32InputRef = {f32Input, f32Input, 0, {6}, {1}};
  MemRefF32 f32KernelRef = {f32Kernel, f32Kernel, 0, {3}, {1}};
  MemRefF32 f32Output;
  _mlir_ciface_f32_correlation(&f32Output, &f32InputRef, &f32KernelRef);
  if (f32Output.sizes[0] != 4)
    return 1;
  for (unsigned output = 0; output < 4; ++output) {
    float actual = f32Output.aligned[f32Output.offset + output * f32Output.strides[0]];
    float expected = f32Correlation(f32Input, f32Kernel, output);
    if (f32Bits(actual) != f32Bits(expected)) {
      fprintf(stderr, "f32 correlation output %u: %08x/%08x\n", output, f32Bits(actual),
              f32Bits(expected));
      free(f32Output.allocated);
      return 1;
    }
  }
  free(f32Output.allocated);
  return 0;
}
