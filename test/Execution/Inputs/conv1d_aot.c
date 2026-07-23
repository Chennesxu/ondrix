#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern int16_t q15_convolution_value(int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                     int16_t, int16_t, int64_t);
extern int16_t q15_correlation_value(int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                     int16_t, int16_t, int64_t);
extern float f32_convolution_value(float, float, float, float, float, float, float, float, float,
                                   int64_t);
extern float f32_correlation_value(float, float, float, float, float, float, float, float, float,
                                   int64_t);

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

static int16_t q15Reference(const int16_t input[6], const int16_t kernel[3], unsigned output,
                            int convolution) {
  int64_t accumulator = 0;
  for (unsigned k = 0; k < 3; ++k) {
    unsigned kernelIndex = convolution ? 2 - k : k;
    accumulator =
        clampI40((__int128)accumulator + (__int128)input[output + k] * kernel[kernelIndex]);
  }
  return exportQ15(accumulator);
}

static float f32Reference(const float input[6], const float kernel[3], unsigned output,
                          int convolution) {
  float accumulator = 0.0f;
  for (unsigned k = 0; k < 3; ++k) {
    unsigned kernelIndex = convolution ? 2 - k : k;
    accumulator = fmaf(input[output + k], kernel[kernelIndex], accumulator);
  }
  return accumulator;
}

static uint32_t f32Bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct Q15Case {
  int16_t input[6];
  int16_t kernel[3];
};

struct F32Case {
  float input[6];
  float kernel[3];
};

int main(void) {
  static const struct Q15Case q15Cases[] = {
      {{32767, -32768, 16384, -8192, 4096, -2048}, {16384, -8192, 4096}},
      {{INT16_MIN, INT16_MIN, INT16_MAX, INT16_MAX, -1, 1}, {INT16_MIN, INT16_MAX, 12345}},
      {{1, 3, 5, 7, 9, 11}, {16384, 8192, -4096}},
  };
  static const struct F32Case f32Cases[] = {
      {{1.25f, -2.0f, 3.5f, -4.25f, 5.0f, -6.5f}, {0.5f, -1.25f, 2.0f}},
      {{INFINITY, -0.0f, 3.0f, NAN, -5.0f, 7.0f}, {0.25f, -2.0f, 1.5f}},
      {{0x1.000002p0f, -0x1.fffffep-1f, 0x1.000004p-2f, 8.0f, -16.0f, 32.0f},
       {0x1.000002p-1f, -0x1.000002p-2f, 0x1.fffffep-1f}},
  };

  for (unsigned caseIndex = 0; caseIndex < sizeof(q15Cases) / sizeof(q15Cases[0]); ++caseIndex) {
    const int16_t *x = q15Cases[caseIndex].input;
    const int16_t *k = q15Cases[caseIndex].kernel;
    for (unsigned output = 0; output < 4; ++output) {
      int16_t convolution =
          q15_convolution_value(x[0], x[1], x[2], x[3], x[4], x[5], k[0], k[1], k[2], output);
      int16_t correlation =
          q15_correlation_value(x[0], x[1], x[2], x[3], x[4], x[5], k[0], k[1], k[2], output);
      int16_t expectedConvolution = q15Reference(x, k, output, 1);
      int16_t expectedCorrelation = q15Reference(x, k, output, 0);
      if (convolution != expectedConvolution || correlation != expectedCorrelation) {
        fprintf(stderr, "Q15 case %u output %u: convolution %d/%d, correlation %d/%d\n", caseIndex,
                output, convolution, expectedConvolution, correlation, expectedCorrelation);
        return 1;
      }
    }
  }

  for (unsigned caseIndex = 0; caseIndex < sizeof(f32Cases) / sizeof(f32Cases[0]); ++caseIndex) {
    const float *x = f32Cases[caseIndex].input;
    const float *k = f32Cases[caseIndex].kernel;
    for (unsigned output = 0; output < 4; ++output) {
      float convolution =
          f32_convolution_value(x[0], x[1], x[2], x[3], x[4], x[5], k[0], k[1], k[2], output);
      float correlation =
          f32_correlation_value(x[0], x[1], x[2], x[3], x[4], x[5], k[0], k[1], k[2], output);
      float expectedConvolution = f32Reference(x, k, output, 1);
      float expectedCorrelation = f32Reference(x, k, output, 0);
      if (!(isnan(convolution) && isnan(expectedConvolution)) &&
          f32Bits(convolution) != f32Bits(expectedConvolution)) {
        fprintf(stderr, "f32 convolution case %u output %u: %08x/%08x\n", caseIndex, output,
                f32Bits(convolution), f32Bits(expectedConvolution));
        return 1;
      }
      if (!(isnan(correlation) && isnan(expectedCorrelation)) &&
          f32Bits(correlation) != f32Bits(expectedCorrelation)) {
        fprintf(stderr, "f32 correlation case %u output %u: %08x/%08x\n", caseIndex, output,
                f32Bits(correlation), f32Bits(expectedCorrelation));
        return 1;
      }
    }
  }
  return 0;
}
