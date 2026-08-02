#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_q15_filtered_spectrum(MemRefI16 *, MemRefI16 *);
/* The same pipeline prefix stopping at the spectrum. Magnitude is not
 * injective - a real/imaginary swap or a sign error can leave every bin's
 * modulus unchanged - so the packed bins are observed directly. This is a
 * stage-isolated compilation of the shared prefix, not an observation of the
 * intermediate the forwarded object deletes. */
extern void _mlir_ciface_q15_filtered_spectrum_stage(MemRefI32 *, MemRefI16 *);

enum { kSignalLength = 72, kTapCount = 9, kExtent = 64, kBinCount = 33, kRandomTrialCount = 64 };

/* Golden Q15 taps for the frozen windowed-sinc lowpass profile
 * (N = 9, fc = 1/4), derived independently with 50-digit mpmath from the
 * real-valued contract equation; they are not read from the compiled
 * module, so a compiler-side design or quantization change breaks this
 * gate. sum(|h|) = 35928, so |accumulator| <= 32768 * 35928 < 2^31 and the
 * i40 saturating accumulator can never clamp: exact int64_t accumulation
 * below is bit-equivalent to the contract. */
static const int16_t kGoldenTaps[kTapCount] = {0, -747, 0, 9025, 16384, 9025, 0, -747, 0};

struct Complex {
  int16_t real;
  int16_t imaginary;
};

/* Forward twiddles w_64^k, quantized to Q15 with round-half-even and
 * declared saturation, derived independently with 50-digit mpmath. */
static const struct Complex kForwardTwiddles64[32] = {
    {32767, 0},       {32610, -3212},   {32138, -6393},   {31357, -9512},   {30274, -12540},
    {28899, -15447},  {27246, -18205},  {25330, -20788},  {23170, -23170},  {20788, -25330},
    {18205, -27246},  {15447, -28899},  {12540, -30274},  {9512, -31357},   {6393, -32138},
    {3212, -32610},   {0, -32768},      {-3212, -32610},  {-6393, -32138},  {-9512, -31357},
    {-12540, -30274}, {-15447, -28899}, {-18205, -27246}, {-20788, -25330}, {-23170, -23170},
    {-25330, -20788}, {-27246, -18205}, {-28899, -15447}, {-30274, -12540}, {-31357, -9512},
    {-32138, -6393},  {-32610, -3212}};

static int16_t requantize(int64_t value, unsigned shift) {
  const int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
    remainder += divisor;
  }
  const int64_t half = divisor >> 1;
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  if (quotient < INT16_MIN)
    return INT16_MIN;
  if (quotient > INT16_MAX)
    return INT16_MAX;
  return (int16_t)quotient;
}

/* Valid-boundary FIR against the golden taps, then the declared
 * round-half-even requantization from the exact product scale back to Q15
 * with saturation. */
static int16_t referenceFilterOutput(const int16_t *input, int64_t index) {
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    accumulator += (int64_t)input[index + tap] * (int64_t)kGoldenTaps[tap];

  int64_t quotient = accumulator / 32768;
  int64_t remainder = accumulator % 32768;
  if (remainder < 0) {
    remainder += 32768;
    --quotient;
  }
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
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

static int32_t toSigned32(uint32_t bits) {
  return bits <= (uint32_t)INT32_MAX ? (int32_t)bits
                                     : (int32_t)((int64_t)bits - (INT64_C(1) << 32));
}

static int32_t pack(struct Complex value) {
  return toSigned32(((uint32_t)(uint16_t)value.imaginary << 16) | (uint32_t)(uint16_t)value.real);
}

static struct Complex unpack(int32_t value) {
  return (struct Complex){decodeSigned16((uint16_t)value),
                          decodeSigned16((uint16_t)((uint32_t)value >> 16))};
}

static void butterfly(int32_t aBits, int32_t bBits, struct Complex w, int32_t *out0,
                      int32_t *out1) {
  struct Complex a = unpack(aBits);
  struct Complex b = unpack(bBits);
  int16_t tr = requantize((int64_t)b.real * w.real - (int64_t)b.imaginary * w.imaginary, 15);
  int16_t ti = requantize((int64_t)b.real * w.imaginary + (int64_t)b.imaginary * w.real, 15);
  *out0 = pack((struct Complex){
      requantize((int64_t)a.real + tr, 1),
      requantize((int64_t)a.imaginary + ti, 1),
  });
  *out1 = pack((struct Complex){
      requantize((int64_t)a.real - tr, 1),
      requantize((int64_t)a.imaginary - ti, 1),
  });
}

static void cfftRecursive(const int32_t *input, unsigned stride, unsigned offset, unsigned n,
                          int32_t *output) {
  if (n == 1) {
    output[0] = input[offset];
    return;
  }
  int32_t even[kExtent / 2];
  int32_t odd[kExtent / 2];
  cfftRecursive(input, stride * 2, offset, n / 2, even);
  cfftRecursive(input, stride * 2, offset + stride, n / 2, odd);
  for (unsigned k = 0; k < n / 2; ++k)
    butterfly(even[k], odd[k], kForwardTwiddles64[k * (kExtent / n)], &output[k],
              &output[k + n / 2]);
}

/* Exact floor square root via a libm estimate plus correction loops, then
 * the declared nearest rounding ((root + 1/2)^2 is never an integer, so
 * there is no tie) and Q15 saturation. Deliberately a different method
 * than the compiled bit-by-bit lowering. */
static int16_t referenceMagnitude(int32_t packedBin) {
  struct Complex bin = unpack(packedBin);
  int64_t sum = (int64_t)bin.real * bin.real + (int64_t)bin.imaginary * bin.imaginary;
  int64_t root = (int64_t)sqrt((double)sum);
  while (root > 0 && root * root > sum)
    --root;
  while ((root + 1) * (root + 1) <= sum)
    ++root;
  if (sum - root * root > root)
    ++root;
  if (root > 32767)
    root = 32767;
  return (int16_t)root;
}

/* The whole four-stage chain, stage by stage and independent of the
 * compiler: design table, valid FIR, staged RFFT, magnitude. */
static void referenceBins(const int16_t *signal, int32_t *bins) {
  int32_t packedFiltered[kExtent];
  for (unsigned i = 0; i < kExtent; ++i)
    packedFiltered[i] = (int32_t)((uint32_t)(uint16_t)referenceFilterOutput(signal, (int64_t)i));

  int32_t spectrum[kExtent];
  cfftRecursive(packedFiltered, 1, 0, kExtent, spectrum);
  spectrum[0] = (int32_t)((uint32_t)spectrum[0] & 0xFFFFu);
  spectrum[kExtent / 2] = (int32_t)((uint32_t)spectrum[kExtent / 2] & 0xFFFFu);

  for (unsigned i = 0; i < kBinCount; ++i)
    bins[i] = spectrum[i];
}

static void referenceSpectrum(const int16_t *signal, int16_t *expected) {
  int32_t bins[kBinCount];
  referenceBins(signal, bins);
  for (unsigned i = 0; i < kBinCount; ++i)
    expected[i] = referenceMagnitude(bins[i]);
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int16_t toSigned16(uint32_t bits) {
  uint32_t low = bits & 0xFFFFu;
  return (int16_t)(low < 32768u ? (int32_t)low : (int32_t)low - 65536);
}

static int checkBins(const int16_t *signal, const char *label) {
  MemRefI16 inputRef = {(int16_t *)signal, (int16_t *)signal, 0, {kSignalLength}, {1}};
  MemRefI32 output;
  _mlir_ciface_q15_filtered_spectrum_stage(&output, &inputRef);

  int32_t expected[kBinCount];
  referenceBins(signal, expected);

  int failed = output.sizes[0] != kBinCount;
  if (failed)
    fprintf(stderr, "%s: spectrum length %lld\n", label, (long long)output.sizes[0]);
  int64_t count = output.sizes[0] < kBinCount ? output.sizes[0] : kBinCount;
  for (int64_t i = 0; i < count; ++i) {
    int32_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s packed bin %lld: got %d, expected %d\n", label, (long long)i, actual,
              expected[i]);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

static int check(const int16_t *signal, const char *label) {
  MemRefI16 inputRef = {(int16_t *)signal, (int16_t *)signal, 0, {kSignalLength}, {1}};
  MemRefI16 output;
  _mlir_ciface_q15_filtered_spectrum(&output, &inputRef);

  int16_t expected[kBinCount];
  referenceSpectrum(signal, expected);

  int failed = checkBins(signal, label);
  if (output.sizes[0] != kBinCount) {
    fprintf(stderr, "%s: output length %lld\n", label, (long long)output.sizes[0]);
    failed = 1;
  }
  int64_t count = output.sizes[0] < kBinCount ? output.sizes[0] : kBinCount;
  for (int64_t i = 0; i < count; ++i) {
    int16_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s bin %lld: got %d, expected %d\n", label, (long long)i, actual,
              expected[i]);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

int main(void) {
  int failed = 0;
  int16_t signal[kSignalLength];

  for (unsigned i = 0; i < kSignalLength; ++i)
    signal[i] = 0;
  failed |= check(signal, "zero");

  for (unsigned i = 0; i < kSignalLength; ++i)
    signal[i] = (i == 0) ? INT16_MAX : 0;
  failed |= check(signal, "impulse 0");

  for (unsigned i = 0; i < kSignalLength; ++i)
    signal[i] = (i == 35) ? INT16_MAX : 0;
  failed |= check(signal, "impulse 35");

  for (unsigned i = 0; i < kSignalLength; ++i)
    signal[i] = 16384;
  failed |= check(signal, "dc positive");

  for (unsigned i = 0; i < kSignalLength; ++i)
    signal[i] = INT16_MIN;
  failed |= check(signal, "dc negative");

  for (unsigned i = 0; i < kSignalLength; ++i)
    signal[i] = (i & 1) ? -INT16_MAX : INT16_MAX;
  failed |= check(signal, "nyquist");

  uint32_t state = 0x5C7B19E3u;
  for (int trial = 0; trial < kRandomTrialCount; ++trial) {
    char label[32];
    for (unsigned i = 0; i < kSignalLength; ++i) {
      state = nextState(state);
      signal[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "trial %d", trial);
    failed |= check(signal, label);
  }
  return failed;
}
