/* Independent reference for the packed-Q31 real-spectrum contract: real i32
 * samples packed into the i64 container, the re-frozen raw-high CFFT core,
 * Hermitian compaction with canonical real endpoints, and the mirror with
 * saturating conjugation at the 32-bit rails. The twiddle tables repeat the
 * frozen 50-digit-mpmath words the CFFT gate pins. */

#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI64;

#ifndef ONDRIX_OX_RFFT_ROUND_TRIP_Q31
extern void _mlir_ciface_rfft8_q31(MemRefI64 *result, MemRefI32 *input);
extern void _mlir_ciface_rfft64_q31(MemRefI64 *result, MemRefI32 *input);
extern void _mlir_ciface_irfft8_q31(MemRefI32 *result, MemRefI64 *input);
extern void _mlir_ciface_irfft64_q31(MemRefI32 *result, MemRefI64 *input);
extern void _mlir_ciface_rfft_round_trip8_q31(MemRefI32 *result, MemRefI32 *input);
#else
/* The .ox binding gate reuses only the round-trip battery under its own
 * kernel symbol. */
extern void _mlir_ciface_q31_rfft_round_trip(MemRefI32 *result, MemRefI32 *input);
#define _mlir_ciface_rfft_round_trip8_q31 _mlir_ciface_q31_rfft_round_trip
#endif

struct Complex {
  int32_t real;
  int32_t imaginary;
};

enum { kMaxExtent = 64 };

static const struct Complex kForwardTwiddles64[32] = {
    {2147483647, 0},
    {2137142927, -210490206},
    {2106220352, -418953276},
    {2055013723, -623381598},
    {1984016189, -821806413},
    {1893911494, -1012316784},
    {1785567396, -1193077991},
    {1660027308, -1362349204},
    {1518500250, -1518500250},
    {1362349204, -1660027308},
    {1193077991, -1785567396},
    {1012316784, -1893911494},
    {821806413, -1984016189},
    {623381598, -2055013723},
    {418953276, -2106220352},
    {210490206, -2137142927},
    {0, INT32_MIN},
    {-210490206, -2137142927},
    {-418953276, -2106220352},
    {-623381598, -2055013723},
    {-821806413, -1984016189},
    {-1012316784, -1893911494},
    {-1193077991, -1785567396},
    {-1362349204, -1660027308},
    {-1518500250, -1518500250},
    {-1660027308, -1362349204},
    {-1785567396, -1193077991},
    {-1893911494, -1012316784},
    {-1984016189, -821806413},
    {-2055013723, -623381598},
    {-2106220352, -418953276},
    {-2137142927, -210490206},
};

static const struct Complex kInverseTwiddles64[32] = {
    {2147483647, 0},
    {2137142927, 210490206},
    {2106220352, 418953276},
    {2055013723, 623381598},
    {1984016189, 821806413},
    {1893911494, 1012316784},
    {1785567396, 1193077991},
    {1660027308, 1362349204},
    {1518500250, 1518500250},
    {1362349204, 1660027308},
    {1193077991, 1785567396},
    {1012316784, 1893911494},
    {821806413, 1984016189},
    {623381598, 2055013723},
    {418953276, 2106220352},
    {210490206, 2137142927},
    {0, INT32_MAX},
    {-210490206, 2137142927},
    {-418953276, 2106220352},
    {-623381598, 2055013723},
    {-821806413, 1984016189},
    {-1012316784, 1893911494},
    {-1193077991, 1785567396},
    {-1362349204, 1660027308},
    {-1518500250, 1518500250},
    {-1660027308, 1362349204},
    {-1785567396, 1193077991},
    {-1893911494, 1012316784},
    {-1984016189, 821806413},
    {-2055013723, 623381598},
    {-2106220352, 418953276},
    {-2137142927, 210490206},
};

static int32_t saturate32(__int128 value) {
  if (value < INT32_MIN)
    return INT32_MIN;
  if (value > INT32_MAX)
    return INT32_MAX;
  return (int32_t)value;
}

static __int128 floorShift(__int128 value, unsigned shift) {
  const __int128 divisor = (__int128)1 << shift;
  __int128 quotient = value / divisor;
  if (value % divisor != 0 && value < 0)
    --quotient;
  return quotient;
}

static int32_t decodeSigned32(uint32_t bits) {
  int64_t value = bits;
  if ((bits & UINT32_C(0x80000000)) != 0)
    value -= INT64_C(4294967296);
  return (int32_t)value;
}

static int64_t toSigned64(uint64_t bits) {
  return bits < UINT64_C(0x8000000000000000)
             ? (int64_t)bits
             : (int64_t)(__int128)((__int128)bits - ((__int128)1 << 64));
}

static int64_t pack(struct Complex value) {
  return toSigned64(((uint64_t)(uint32_t)value.imaginary << 32) | (uint64_t)(uint32_t)value.real);
}

static struct Complex unpack(int64_t value) {
  return (struct Complex){decodeSigned32((uint32_t)value),
                          decodeSigned32((uint32_t)((uint64_t)value >> 32))};
}

static int32_t saturatingNegate(int32_t value) { return value == INT32_MIN ? INT32_MAX : -value; }

static int64_t canonicalReal(int64_t value) {
  return pack((struct Complex){unpack(value).real, 0});
}

static int64_t conjugate(int64_t value) {
  struct Complex unpacked = unpack(value);
  return pack((struct Complex){unpacked.real, saturatingNegate(unpacked.imaginary)});
}

/* The raw-high butterfly: per-term floors, exact combine, saturating doubling,
 * floor-by-one saturating stage outputs. */
static void butterfly(int64_t aBits, int64_t bBits, struct Complex w, int64_t *out0,
                      int64_t *out1) {
  struct Complex a = unpack(aBits);
  struct Complex b = unpack(bBits);
  int64_t termReal = (int64_t)floorShift((__int128)b.real * w.real, 32) -
                     (int64_t)floorShift((__int128)b.imaginary * w.imaginary, 32);
  int64_t termImaginary = (int64_t)floorShift((__int128)b.real * w.imaginary, 32) +
                          (int64_t)floorShift((__int128)b.imaginary * w.real, 32);
  int32_t tr = saturate32((__int128)termReal * 2);
  int32_t ti = saturate32((__int128)termImaginary * 2);
  *out0 = pack((struct Complex){
      saturate32(floorShift((__int128)a.real + tr, 1)),
      saturate32(floorShift((__int128)a.imaginary + ti, 1)),
  });
  *out1 = pack((struct Complex){
      saturate32(floorShift((__int128)a.real - tr, 1)),
      saturate32(floorShift((__int128)a.imaginary - ti, 1)),
  });
}

static void cfftRecursive(const int64_t *input, unsigned stride, unsigned offset, unsigned n,
                          const struct Complex *twiddles, int64_t *output) {
  if (n == 1) {
    output[0] = input[offset];
    return;
  }
  int64_t even[kMaxExtent];
  int64_t odd[kMaxExtent];
  cfftRecursive(input, stride * 2, offset, n / 2, twiddles, even);
  cfftRecursive(input, stride * 2, offset + stride, n / 2, twiddles, odd);
  for (unsigned k = 0; k < n / 2; ++k)
    butterfly(even[k], odd[k], twiddles[k * (kMaxExtent / n)], &output[k], &output[k + n / 2]);
}

static void referenceRfft(const int32_t *input, unsigned size, int64_t *output) {
  int64_t complexInput[kMaxExtent];
  int64_t transformed[kMaxExtent];
  for (unsigned i = 0; i < size; ++i)
    complexInput[i] = pack((struct Complex){input[i], 0});
  cfftRecursive(complexInput, 1, 0, size, kForwardTwiddles64, transformed);
  for (unsigned i = 0; i <= size / 2; ++i)
    output[i] = transformed[i];
  output[0] = canonicalReal(output[0]);
  output[size / 2] = canonicalReal(output[size / 2]);
}

static void referenceIrfft(const int64_t *input, unsigned size, int32_t *output) {
  int64_t spectrum[kMaxExtent];
  int64_t transformed[kMaxExtent];
  unsigned half = size / 2;
  spectrum[0] = canonicalReal(input[0]);
  spectrum[half] = canonicalReal(input[half]);
  for (unsigned k = 1; k < half; ++k) {
    spectrum[k] = input[k];
    spectrum[size - k] = conjugate(input[k]);
  }
  cfftRecursive(spectrum, 1, 0, size, kInverseTwiddles64, transformed);
  for (unsigned i = 0; i < size; ++i)
    output[i] = unpack(transformed[i]).real;
}

#ifndef ONDRIX_OX_RFFT_ROUND_TRIP_Q31
static bool checkForward(const char *name, const int32_t *input, unsigned size) {
  int64_t expected[kMaxExtent / 2 + 1];
  referenceRfft(input, size, expected);

  MemRefI32 inputRef = {(int32_t *)input, (int32_t *)input, 0, {(int64_t)size}, {1}};
  MemRefI64 output;
  if (size == 8)
    _mlir_ciface_rfft8_q31(&output, &inputRef);
  else
    _mlir_ciface_rfft64_q31(&output, &inputRef);
  bool passed = output.sizes[0] == (int64_t)(size / 2 + 1);
  if (!passed)
    fprintf(stderr, "%s: expected extent %u, got %" PRId64 "\n", name, size / 2 + 1,
            output.sizes[0]);
  for (unsigned i = 0; passed && i <= size / 2; ++i) {
    int64_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s forward[%u]: expected %016" PRIx64 ", got %016" PRIx64 "\n", name, i,
              (uint64_t)expected[i], (uint64_t)actual);
      passed = false;
    }
  }
  free(output.allocated);
  return passed;
}

static bool checkInverse(const char *name, const int64_t *spectrum, unsigned size) {
  int32_t expected[kMaxExtent];
  referenceIrfft(spectrum, size, expected);

  MemRefI64 inputRef = {
      (int64_t *)spectrum, (int64_t *)spectrum, 0, {(int64_t)(size / 2 + 1)}, {1}};
  MemRefI32 output;
  if (size == 8)
    _mlir_ciface_irfft8_q31(&output, &inputRef);
  else
    _mlir_ciface_irfft64_q31(&output, &inputRef);
  bool passed = output.sizes[0] == (int64_t)size;
  if (!passed)
    fprintf(stderr, "%s: expected extent %u, got %" PRId64 "\n", name, size, output.sizes[0]);
  for (unsigned i = 0; passed && i < size; ++i) {
    int32_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s inverse[%u]: expected %08" PRIx32 ", got %08" PRIx32 "\n", name, i,
              (uint32_t)expected[i], (uint32_t)actual);
      passed = false;
    }
  }
  free(output.allocated);
  return passed;
}
#endif

static bool checkRoundTrip8(const char *name, const int32_t *input) {
  int64_t spectrum[5];
  int32_t expected[8];
  referenceRfft(input, 8, spectrum);
  referenceIrfft(spectrum, 8, expected);

  MemRefI32 inputRef = {(int32_t *)input, (int32_t *)input, 0, {8}, {1}};
  MemRefI32 output;
  _mlir_ciface_rfft_round_trip8_q31(&output, &inputRef);
  bool passed = output.sizes[0] == 8;
  for (unsigned i = 0; passed && i < 8; ++i) {
    int32_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s round-trip[%u]: expected %08" PRIx32 ", got %08" PRIx32 "\n", name, i,
              (uint32_t)expected[i], (uint32_t)actual);
      passed = false;
    }
  }
  free(output.allocated);
  return passed;
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

int main(void) {
  static const int32_t cases8[][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},
      {INT32_MAX, 0, 0, 0, 0, 0, 0, 0},
      {1200000001, -1600000000, INT32_MIN, 900000000, -400000007, 2000000000, -2100000000,
       1400000000},
      {INT32_MAX, INT32_MIN, INT32_MAX, INT32_MIN, INT32_MAX, INT32_MIN, INT32_MAX, INT32_MIN},
      {1, -3, 5, -7, 9, -11, 13, -15},
  };
  /* An arbitrary Hermitian-compact spectrum whose interior bins carry the
   * imaginary rail, so the mirror's saturating conjugation of INT32_MIN is
   * executed, plus odd words that keep every stage floor active. */
  static const struct Complex spectrum8[5] = {
      {INT32_MIN, 12001}, {INT32_MIN, INT32_MIN}, {1111111111, INT32_MIN},
      {-2000000001, 3},   {INT32_MAX, -5},
  };

  int failed = 0;
  for (unsigned i = 0; i < sizeof(cases8) / sizeof(cases8[0]); ++i) {
    char name[24];
    snprintf(name, sizeof(name), "rfft8-case-%u", i);
#ifndef ONDRIX_OX_RFFT_ROUND_TRIP_Q31
    failed |= !checkForward(name, cases8[i], 8);
#endif
    failed |= !checkRoundTrip8(name, cases8[i]);
  }

#ifndef ONDRIX_OX_RFFT_ROUND_TRIP_Q31
  int64_t packedSpectrum8[5];
  for (unsigned i = 0; i < 5; ++i)
    packedSpectrum8[i] = pack(spectrum8[i]);
  failed |= !checkInverse("arbitrary-spectrum8", packedSpectrum8, 8);

  /* The 64-point ceiling reaches the deep frozen table rows in both
   * directions; the spectrum for the inverse leg is xorshift breadth with the
   * conjugation rail planted in one interior bin. */
  int32_t input64[kMaxExtent];
  uint32_t state = 0x2545f491u;
  for (unsigned i = 0; i < kMaxExtent; ++i) {
    state = nextState(state);
    input64[i] = decodeSigned32(state);
  }
  failed |= !checkForward("rfft64-xorshift", input64, kMaxExtent);

  int64_t spectrum64[kMaxExtent / 2 + 1];
  for (unsigned i = 0; i <= kMaxExtent / 2; ++i) {
    state = nextState(state);
    int32_t real = decodeSigned32(state);
    state = nextState(state);
    spectrum64[i] = pack((struct Complex){real, decodeSigned32(state)});
  }
  spectrum64[17] = pack((struct Complex){-77777777, INT32_MIN});
  failed |= !checkInverse("irfft64-xorshift", spectrum64, kMaxExtent);
#endif

  return failed;
}
