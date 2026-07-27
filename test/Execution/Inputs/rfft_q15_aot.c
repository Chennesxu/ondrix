#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef ONDRIX_OX_RFFT_ROUND_TRIP
extern int32_t rfft8_q15_value(int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                               int16_t, int64_t);
extern int32_t rfft16_q15_value(int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                int16_t, int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                int16_t, int16_t, int64_t);
extern int16_t irfft8_q15_value(int32_t, int32_t, int32_t, int32_t, int32_t, int64_t);
extern int16_t irfft16_q15_value(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                 int32_t, int32_t, int64_t);
extern int16_t rfft_round_trip8_q15_value(int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                          int16_t, int16_t, int64_t);
extern int16_t rfft_round_trip16_q15_value(int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                           int16_t, int16_t, int16_t, int16_t, int16_t, int16_t,
                                           int16_t, int16_t, int16_t, int16_t, int64_t);
#else
typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

extern void _mlir_ciface_q15_rfft_round_trip(MemRefI16 *result, MemRefI16 *input);
#endif

struct Complex {
  int16_t real;
  int16_t imaginary;
};

static int16_t requantize(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  __int128 remainder = value % divisor;
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

static int16_t decodeSigned16(uint16_t bits) {
  int32_t value = bits;
  if ((bits & UINT16_C(0x8000)) != 0)
    value -= INT32_C(65536);
  return (int16_t)value;
}

static int32_t pack(struct Complex value) {
  return (int32_t)(((uint32_t)(uint16_t)value.imaginary << 16) | (uint32_t)(uint16_t)value.real);
}

static struct Complex unpack(int32_t value) {
  return (struct Complex){decodeSigned16((uint16_t)value),
                          decodeSigned16((uint16_t)((uint32_t)value >> 16))};
}

static int16_t saturatingNegate(int16_t value) {
  return value == INT16_MIN ? INT16_MAX : (int16_t)-value;
}

static int32_t canonicalReal(int32_t value) {
  return pack((struct Complex){unpack(value).real, 0});
}

static int32_t conjugate(int32_t value) {
  struct Complex unpacked = unpack(value);
  return pack((struct Complex){unpacked.real, saturatingNegate(unpacked.imaginary)});
}

static void butterfly(int32_t aBits, int32_t bBits, int32_t twiddleBits, int32_t *out0,
                      int32_t *out1) {
  struct Complex a = unpack(aBits);
  struct Complex b = unpack(bBits);
  struct Complex w = unpack(twiddleBits);
  int16_t tr = requantize((__int128)b.real * w.real - (__int128)b.imaginary * w.imaginary, 15);
  int16_t ti = requantize((__int128)b.real * w.imaginary + (__int128)b.imaginary * w.real, 15);
  *out0 = pack((struct Complex){
      requantize((__int128)a.real + tr, 1),
      requantize((__int128)a.imaginary + ti, 1),
  });
  *out1 = pack((struct Complex){
      requantize((__int128)a.real - tr, 1),
      requantize((__int128)a.imaginary - ti, 1),
  });
}

static struct Complex twiddle(unsigned size, unsigned index, bool inverse) {
  static const struct Complex forward16[8] = {
      {INT16_MAX, 0}, {30274, -12540},  {23170, -23170},  {12540, -30274},
      {0, INT16_MIN}, {-12540, -30274}, {-23170, -23170}, {-30274, -12540},
  };
  static const struct Complex inverse16[8] = {
      {INT16_MAX, 0}, {30274, 12540},  {23170, 23170},  {12540, 30274},
      {0, INT16_MAX}, {-12540, 30274}, {-23170, 23170}, {-30274, 12540},
  };
  unsigned stride = 16 / size;
  return (inverse ? inverse16 : forward16)[index * stride];
}

static void transform(const int32_t *input, unsigned size, bool inverse, int32_t *output) {
  if (size == 1) {
    output[0] = input[0];
    return;
  }

  int32_t evenInput[8];
  int32_t oddInput[8];
  int32_t evenOutput[8];
  int32_t oddOutput[8];
  unsigned half = size / 2;
  for (unsigned i = 0; i < half; ++i) {
    evenInput[i] = input[2 * i];
    oddInput[i] = input[2 * i + 1];
  }
  transform(evenInput, half, inverse, evenOutput);
  transform(oddInput, half, inverse, oddOutput);
  for (unsigned k = 0; k < half; ++k)
    butterfly(evenOutput[k], oddOutput[k], pack(twiddle(size, k, inverse)), &output[k],
              &output[k + half]);
}

static void referenceRfft(const int16_t *input, unsigned size, int32_t *output) {
  int32_t complexInput[16];
  int32_t transformed[16];
  for (unsigned i = 0; i < size; ++i)
    complexInput[i] = pack((struct Complex){input[i], 0});
  transform(complexInput, size, false, transformed);
  for (unsigned i = 0; i <= size / 2; ++i)
    output[i] = transformed[i];
  output[0] = canonicalReal(output[0]);
  output[size / 2] = canonicalReal(output[size / 2]);
}

static void referenceIrfft(const int32_t *input, unsigned size, int16_t *output) {
  int32_t spectrum[16] = {0};
  int32_t transformed[16];
  unsigned half = size / 2;
  spectrum[0] = canonicalReal(input[0]);
  spectrum[half] = canonicalReal(input[half]);
  for (unsigned k = 1; k < half; ++k) {
    spectrum[k] = input[k];
    spectrum[size - k] = conjugate(input[k]);
  }
  transform(spectrum, size, true, transformed);
  for (unsigned i = 0; i < size; ++i)
    output[i] = unpack(transformed[i]).real;
}

#ifndef ONDRIX_OX_RFFT_ROUND_TRIP
static int32_t callRfft8(const int16_t *x, unsigned index) {
  return rfft8_q15_value(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], (int64_t)index);
}

static int32_t callRfft16(const int16_t *x, unsigned index) {
  return rfft16_q15_value(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11],
                          x[12], x[13], x[14], x[15], (int64_t)index);
}

static int16_t callIrfft8(const int32_t *x, unsigned index) {
  return irfft8_q15_value(x[0], x[1], x[2], x[3], x[4], (int64_t)index);
}

static int16_t callIrfft16(const int32_t *x, unsigned index) {
  return irfft16_q15_value(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], (int64_t)index);
}

static int16_t callRoundTrip8(const int16_t *x, unsigned index) {
  return rfft_round_trip8_q15_value(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], (int64_t)index);
}

static int16_t callRoundTrip16(const int16_t *x, unsigned index) {
  return rfft_round_trip16_q15_value(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
                                     x[10], x[11], x[12], x[13], x[14], x[15], (int64_t)index);
}

static bool checkForward(const char *name, const int16_t *input, unsigned size) {
  int32_t expected[9];
  referenceRfft(input, size, expected);
  for (unsigned i = 0; i <= size / 2; ++i) {
    int32_t actual = size == 8 ? callRfft8(input, i) : callRfft16(input, i);
    if (actual != expected[i]) {
      fprintf(stderr, "%s forward[%u]: expected %08" PRIx32 ", got %08" PRIx32 "\n", name, i,
              (uint32_t)expected[i], (uint32_t)actual);
      return false;
    }
  }
  return true;
}

static bool checkInverseAndRoundTrip(const char *name, const int16_t *input, unsigned size) {
  int32_t spectrum[9];
  int16_t expected[16];
  referenceRfft(input, size, spectrum);
  referenceIrfft(spectrum, size, expected);

  for (unsigned i = 0; i < size; ++i) {
    int16_t inverseActual = size == 8 ? callIrfft8(spectrum, i) : callIrfft16(spectrum, i);
    int16_t roundTripActual = size == 8 ? callRoundTrip8(input, i) : callRoundTrip16(input, i);
    if (inverseActual != expected[i] || roundTripActual != expected[i]) {
      fprintf(stderr,
              "%s inverse[%u]: expected %" PRId16 ", direct %" PRId16 ", round-trip %" PRId16 "\n",
              name, i, expected[i], inverseActual, roundTripActual);
      return false;
    }
  }
  return true;
}

static bool checkArbitraryInverse(const char *name, const int32_t *spectrum, unsigned size) {
  int16_t expected[16];
  referenceIrfft(spectrum, size, expected);
  for (unsigned i = 0; i < size; ++i) {
    int16_t actual = size == 8 ? callIrfft8(spectrum, i) : callIrfft16(spectrum, i);
    if (actual != expected[i]) {
      fprintf(stderr, "%s[%u]: expected %" PRId16 ", got %" PRId16 "\n", name, i, expected[i],
              actual);
      return false;
    }
  }
  return true;
}

int main(void) {
  static const int32_t arbitrarySpectrum8[5] = {
      (int32_t)UINT32_C(0x80002ee0), (int32_t)UINT32_C(0x8000e4a8), (int32_t)UINT32_C(0x42682328),
      (int32_t)UINT32_C(0xd5088ad0), (int32_t)UINT32_C(0x7fff1388),
  };
  static const int32_t arbitrarySpectrum16[9] = {
      (int32_t)UINT32_C(0x7fff8000), (int32_t)UINT32_C(0x800004d2), (int32_t)UINT32_C(0x4268e4a8),
      (int32_t)UINT32_C(0xd5087530), (int32_t)UINT32_C(0x00017fff), (int32_t)UINT32_C(0xffff0001),
      (int32_t)UINT32_C(0x3039cfc7), (int32_t)UINT32_C(0x8ad07530), (int32_t)UINT32_C(0x80001388),
  };
  static const int16_t cases8[][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {INT16_MAX, 0, 0, 0, 0, 0, 0, 0},
      {12000, -16000, INT16_MIN, 9000, -4000, 30000, -22000, 14000},
      {INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN},
      {1, -3, 5, -7, 9, -11, 13, -15},
  };
  static const int16_t cases16[][16] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {INT16_MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {12000, -16000, INT16_MIN, 9000, -4000, 30000, -22000, 14000, 7000, -25000, 31000, -1, 3, -5,
       17, -32760},
      {INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN,
       INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN},
      {1, -3, 5, -7, 9, -11, 13, -15, 17, -19, 21, -23, 25, -27, 29, -31},
  };

  for (unsigned i = 0; i < sizeof(cases8) / sizeof(cases8[0]); ++i) {
    char name[24];
    snprintf(name, sizeof(name), "rfft8-case-%u", i);
    if (!checkForward(name, cases8[i], 8) || !checkInverseAndRoundTrip(name, cases8[i], 8))
      return 1;
  }
  for (unsigned i = 0; i < sizeof(cases16) / sizeof(cases16[0]); ++i) {
    char name[24];
    snprintf(name, sizeof(name), "rfft16-case-%u", i);
    if (!checkForward(name, cases16[i], 16) || !checkInverseAndRoundTrip(name, cases16[i], 16))
      return 1;
  }
  if (!checkArbitraryInverse("arbitrary-spectrum8", arbitrarySpectrum8, 8) ||
      !checkArbitraryInverse("arbitrary-spectrum16", arbitrarySpectrum16, 16))
    return 1;
  return 0;
}
#else
int main(void) {
  static const int16_t cases[][16] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {INT16_MAX, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {12000, -16000, INT16_MIN, 9000, -4000, 30000, -22000, 14000, 7000, -25000, 31000, -1, 3, -5,
       17, -32760},
      {INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN,
       INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN},
      {1, -3, 5, -7, 9, -11, 13, -15, 17, -19, 21, -23, 25, -27, 29, -31},
  };

  for (unsigned caseIndex = 0; caseIndex < sizeof(cases) / sizeof(cases[0]); ++caseIndex) {
    int32_t spectrum[9];
    int16_t expected[16];
    referenceRfft(cases[caseIndex], 16, spectrum);
    referenceIrfft(spectrum, 16, expected);

    MemRefI16 input = {(int16_t *)cases[caseIndex], (int16_t *)cases[caseIndex], 0, {16}, {1}};
    MemRefI16 output;
    _mlir_ciface_q15_rfft_round_trip(&output, &input);
    if (output.sizes[0] != 16) {
      fprintf(stderr, "round-trip case %u: expected extent 16, got %" PRId64 "\n", caseIndex,
              output.sizes[0]);
      free(output.allocated);
      return 1;
    }
    for (unsigned i = 0; i < 16; ++i) {
      int16_t actual = output.aligned[output.offset + i * output.strides[0]];
      if (actual != expected[i]) {
        fprintf(stderr, "round-trip case %u[%u]: expected %" PRId16 ", got %" PRId16 "\n",
                caseIndex, i, expected[i], actual);
        free(output.allocated);
        return 1;
      }
    }
    free(output.allocated);
  }
  return 0;
}
#endif
