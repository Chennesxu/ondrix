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

extern void _mlir_ciface_rfft256_q15(MemRefI32 *, MemRefI16 *);

enum { kExtent = 256, kBinCount = 129, kTrialCount = 6 };

struct Complex {
  int16_t real;
  int16_t imaginary;
};

/* Forward twiddles w_256^k, quantized to Q15 with round-half-even and
 * declared saturation, derived independently with 50-digit mpmath from the
 * contract equation. A stage of size n uses table[k * (256 / n)]. */
static const struct Complex kForwardTwiddles256[128] = {
    {32767, 0},       {32758, -804},    {32729, -1608},   {32679, -2411},   {32610, -3212},
    {32522, -4011},   {32413, -4808},   {32286, -5602},   {32138, -6393},   {31972, -7180},
    {31786, -7962},   {31581, -8740},   {31357, -9512},   {31114, -10279},  {30853, -11039},
    {30572, -11793},  {30274, -12540},  {29957, -13279},  {29622, -14010},  {29269, -14733},
    {28899, -15447},  {28511, -16151},  {28106, -16846},  {27684, -17531},  {27246, -18205},
    {26791, -18868},  {26320, -19520},  {25833, -20160},  {25330, -20788},  {24812, -21403},
    {24279, -22006},  {23732, -22595},  {23170, -23170},  {22595, -23732},  {22006, -24279},
    {21403, -24812},  {20788, -25330},  {20160, -25833},  {19520, -26320},  {18868, -26791},
    {18205, -27246},  {17531, -27684},  {16846, -28106},  {16151, -28511},  {15447, -28899},
    {14733, -29269},  {14010, -29622},  {13279, -29957},  {12540, -30274},  {11793, -30572},
    {11039, -30853},  {10279, -31114},  {9512, -31357},   {8740, -31581},   {7962, -31786},
    {7180, -31972},   {6393, -32138},   {5602, -32286},   {4808, -32413},   {4011, -32522},
    {3212, -32610},   {2411, -32679},   {1608, -32729},   {804, -32758},    {0, -32768},
    {-804, -32758},   {-1608, -32729},  {-2411, -32679},  {-3212, -32610},  {-4011, -32522},
    {-4808, -32413},  {-5602, -32286},  {-6393, -32138},  {-7180, -31972},  {-7962, -31786},
    {-8740, -31581},  {-9512, -31357},  {-10279, -31114}, {-11039, -30853}, {-11793, -30572},
    {-12540, -30274}, {-13279, -29957}, {-14010, -29622}, {-14733, -29269}, {-15447, -28899},
    {-16151, -28511}, {-16846, -28106}, {-17531, -27684}, {-18205, -27246}, {-18868, -26791},
    {-19520, -26320}, {-20160, -25833}, {-20788, -25330}, {-21403, -24812}, {-22006, -24279},
    {-22595, -23732}, {-23170, -23170}, {-23732, -22595}, {-24279, -22006}, {-24812, -21403},
    {-25330, -20788}, {-25833, -20160}, {-26320, -19520}, {-26791, -18868}, {-27246, -18205},
    {-27684, -17531}, {-28106, -16846}, {-28511, -16151}, {-28899, -15447}, {-29269, -14733},
    {-29622, -14010}, {-29957, -13279}, {-30274, -12540}, {-30572, -11793}, {-30853, -11039},
    {-31114, -10279}, {-31357, -9512},  {-31581, -8740},  {-31786, -7962},  {-31972, -7180},
    {-32138, -6393},  {-32286, -5602},  {-32413, -4808},  {-32522, -4011},  {-32610, -3212},
    {-32679, -2411},  {-32729, -1608},  {-32758, -804}};

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
    butterfly(even[k], odd[k], kForwardTwiddles256[k * (kExtent / n)], &output[k],
              &output[k + n / 2]);
}

/* Real bins 0 and N/2 canonicalize to the zero-extended low half, matching
 * the compact-endpoint contract. */
static int32_t canonicalizeRealBin(int32_t packed) { return (int32_t)((uint32_t)packed & 0xFFFFu); }

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

static int check(const int16_t *input, const char *label) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {kExtent}, {1}};
  MemRefI32 output;
  _mlir_ciface_rfft256_q15(&output, &inputRef);

  int32_t spectrum[kExtent];
  int32_t packedInput[kExtent];
  for (unsigned i = 0; i < kExtent; ++i)
    packedInput[i] = (int32_t)((uint32_t)(uint16_t)input[i]);
  cfftRecursive(packedInput, 1, 0, kExtent, spectrum);
  spectrum[0] = canonicalizeRealBin(spectrum[0]);
  spectrum[kExtent / 2] = canonicalizeRealBin(spectrum[kExtent / 2]);

  int failed = output.sizes[0] != kBinCount;
  int64_t count = output.sizes[0] < kBinCount ? output.sizes[0] : kBinCount;
  for (int64_t i = 0; i < count; ++i) {
    int32_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != spectrum[i]) {
      fprintf(stderr, "%s bin %lld: got %08x, expected %08x\n", label, (long long)i,
              (unsigned)actual, (unsigned)spectrum[i]);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

int main(void) {
  int failed = 0;
  uint32_t state = 0x5EC7A256u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int16_t input[kExtent];
    char label[32];
    for (unsigned i = 0; i < kExtent; ++i) {
      state = nextState(state);
      input[i] = toSigned16(state);
    }
    if (trial == 0)
      for (unsigned i = 0; i < kExtent; ++i)
        input[i] = (i & 1) ? INT16_MIN : INT16_MAX;
    snprintf(label, sizeof label, "trial %d", trial);
    failed |= check(input, label);
  }

  /* Directed corpus: a full-scale impulse at EVERY input position walks
   * energy through every stage/twiddle index pair of the recursion (signs
   * alternate so both saturating directions are exercised), plus DC rails
   * and a single off-center extreme sample. */
  int16_t directed[kExtent];
  for (unsigned position = 0; position < kExtent; ++position) {
    char label[40];
    for (unsigned i = 0; i < kExtent; ++i)
      directed[i] = 0;
    directed[position] = (position & 1) ? INT16_MIN : INT16_MAX;
    snprintf(label, sizeof label, "impulse %u", position);
    failed |= check(directed, label);
  }
  for (unsigned i = 0; i < kExtent; ++i)
    directed[i] = INT16_MAX;
  failed |= check(directed, "dc max");
  for (unsigned i = 0; i < kExtent; ++i)
    directed[i] = INT16_MIN;
  failed |= check(directed, "dc min");
  for (unsigned i = 0; i < kExtent; ++i)
    directed[i] = 0;
  directed[kExtent / 2 + 1] = INT16_MIN;
  failed |= check(directed, "single extreme");
  return failed;
}
