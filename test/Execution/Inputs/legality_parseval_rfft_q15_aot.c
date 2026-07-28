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

extern void _mlir_ciface_parseval_rfft_q15(MemRefI32 *, MemRefI16 *);

enum { kExtent = 64, kBinCount = 33, kTrialCount = 8 };

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

/* divergent selects whether this witness must break or must satisfy the
 * rescaled identity; when goldenTime is nonzero the exact energies of the
 * witness are also pinned so reference drift is caught. */
static int check(const int16_t *input, const char *label, int divergent, int64_t goldenTime,
                 int64_t goldenRescaled) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {kExtent}, {1}};
  MemRefI32 output;
  _mlir_ciface_parseval_rfft_q15(&output, &inputRef);

  /* Reference bins through the frozen contract recursion. */
  int32_t spectrum[kExtent];
  int32_t packedInput[kExtent];
  for (unsigned i = 0; i < kExtent; ++i)
    packedInput[i] = (int32_t)((uint32_t)(uint16_t)input[i]);
  cfftRecursive(packedInput, 1, 0, kExtent, spectrum);
  spectrum[0] = (int32_t)((uint32_t)spectrum[0] & 0xFFFFu);
  spectrum[kExtent / 2] = (int32_t)((uint32_t)spectrum[kExtent / 2] & 0xFFFFu);

  int failed = output.sizes[0] != kBinCount;
  int64_t count = output.sizes[0] < kBinCount ? output.sizes[0] : kBinCount;
  for (int64_t i = 0; i < count; ++i) {
    int32_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != spectrum[i]) {
      fprintf(stderr, "%s bin %lld: got %ld, expected %ld\n", label, (long long)i, (long)actual,
              (long)spectrum[i]);
      failed = 1;
    }
  }
  free(output.allocated);
  if (failed)
    return failed;

  /* Exact time-domain energy (no quantization anywhere). */
  int64_t timeEnergy = 0;
  for (unsigned i = 0; i < kExtent; ++i)
    timeEnergy += (int64_t)input[i] * input[i];

  /* Exact Hermitian spectral energy straight from the raw bin components;
   * no square root, no i16 magnitude saturation, no re-squaring of a
   * rounded value. Bins 1..31 appear twice in the full spectrum. Each bin
   * carries the cumulative 1/64 staged scale, so Parseval over the reals
   * predicts timeEnergy == 64 * spectralEnergy exactly. */
  int64_t spectralEnergy = 0;
  for (unsigned k = 0; k < kBinCount; ++k) {
    struct Complex bin = unpack(spectrum[k]);
    int64_t term = (int64_t)bin.real * bin.real + (int64_t)bin.imaginary * bin.imaginary;
    spectralEnergy += (k == 0 || k == kExtent / 2) ? term : 2 * term;
  }
  int64_t rescaled = 64 * spectralEnergy;

  if (divergent && rescaled == timeEnergy) {
    fprintf(stderr, "%s: identity unexpectedly exact (%lld)\n", label, (long long)timeEnergy);
    return 1;
  }
  if (!divergent && rescaled != timeEnergy) {
    fprintf(stderr, "%s: expected exact identity, got %lld vs %lld\n", label, (long long)timeEnergy,
            (long long)rescaled);
    return 1;
  }
  if (goldenTime != 0 && (timeEnergy != goldenTime || rescaled != goldenRescaled)) {
    fprintf(stderr, "%s: energies %lld/%lld, pinned %lld/%lld\n", label, (long long)timeEnergy,
            (long long)rescaled, (long long)goldenTime, (long long)goldenRescaled);
    return 1;
  }
  return 0;
}

int main(void) {
  int failed = 0;

  /* Benign witness: period-4 input whose components are exact multiples
   * of the staged scales, so every RFFT requantization is exact and the
   * rescaled identity HOLDS bit-wise. Together with the diverging
   * witnesses below this pins the failure as input-dependent — the
   * rewrite is not merely wrong by a constant factor. */
  int16_t benign[kExtent];
  for (unsigned i = 0; i < kExtent; ++i)
    benign[i] = (int16_t)((i % 4) * 2048 - 3072);
  failed |= check(benign, "benign", 0, INT64_C(335544320), INT64_C(335544320));

  /* Full-scale alternating witness with pinned golden energies. */
  int16_t extreme[kExtent];
  for (unsigned i = 0; i < kExtent; ++i)
    extreme[i] = (i & 1) ? INT16_MIN : INT16_MAX;
  failed |= check(extreme, "extreme", 1, INT64_C(68717379616), INT64_C(68711088384));

  uint32_t state = 0x9A25E7A1u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t input[kExtent];
    char label[32];
    for (unsigned i = 0; i < kExtent; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "trial %d", trial);
    failed |= check(input, label, 1, 0, 0);
  }
  return failed;
}
