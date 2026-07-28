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

extern void _mlir_ciface_cfft64_forward_q15(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_cfft64_inverse_q15(MemRefI32 *, MemRefI32 *);

enum { kExtent = 64, kTrialCount = 8 };

struct Complex {
  int16_t real;
  int16_t imaginary;
};

/* Twiddles w_64^k = cos(2*pi*k/64) -+ j*sin(2*pi*k/64), quantized to Q15
 * with round-half-even and declared saturation, derived independently with
 * 50-digit mpmath from the contract equation. A stage of size n uses
 * table[k * (64 / n)]. */
static const struct Complex kForwardTwiddles64[32] = {
    {32767, 0},       {32610, -3212},   {32138, -6393},   {31357, -9512},   {30274, -12540},
    {28899, -15447},  {27246, -18205},  {25330, -20788},  {23170, -23170},  {20788, -25330},
    {18205, -27246},  {15447, -28899},  {12540, -30274},  {9512, -31357},   {6393, -32138},
    {3212, -32610},   {0, -32768},      {-3212, -32610},  {-6393, -32138},  {-9512, -31357},
    {-12540, -30274}, {-15447, -28899}, {-18205, -27246}, {-20788, -25330}, {-23170, -23170},
    {-25330, -20788}, {-27246, -18205}, {-28899, -15447}, {-30274, -12540}, {-31357, -9512},
    {-32138, -6393},  {-32610, -3212}};
static const struct Complex kInverseTwiddles64[32] = {
    {32767, 0},      {32610, 3212},   {32138, 6393},   {31357, 9512},   {30274, 12540},
    {28899, 15447},  {27246, 18205},  {25330, 20788},  {23170, 23170},  {20788, 25330},
    {18205, 27246},  {15447, 28899},  {12540, 30274},  {9512, 31357},   {6393, 32138},
    {3212, 32610},   {0, 32767},      {-3212, 32610},  {-6393, 32138},  {-9512, 31357},
    {-12540, 30274}, {-15447, 28899}, {-18205, 27246}, {-20788, 25330}, {-23170, 23170},
    {-25330, 20788}, {-27246, 18205}, {-28899, 15447}, {-30274, 12540}, {-31357, 9512},
    {-32138, 6393},  {-32610, 3212}};

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
  return (int32_t)(bits < 0x80000000u ? (int64_t)bits : (int64_t)bits - 4294967296);
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
                          const struct Complex *twiddles, int32_t *output) {
  if (n == 1) {
    output[0] = input[offset];
    return;
  }
  int32_t even[kExtent / 2];
  int32_t odd[kExtent / 2];
  cfftRecursive(input, stride * 2, offset, n / 2, twiddles, even);
  cfftRecursive(input, stride * 2, offset + stride, n / 2, twiddles, odd);
  for (unsigned k = 0; k < n / 2; ++k)
    butterfly(even[k], odd[k], twiddles[k * (kExtent / n)], &output[k], &output[k + n / 2]);
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int check(void (*kernel)(MemRefI32 *, MemRefI32 *), const struct Complex *twiddles,
                 const int32_t *input, const char *label) {
  MemRefI32 inputRef = {(int32_t *)input, (int32_t *)input, 0, {kExtent}, {1}};
  MemRefI32 output;
  kernel(&output, &inputRef);

  int32_t expected[kExtent];
  cfftRecursive(input, 1, 0, kExtent, twiddles, expected);

  int failed = output.sizes[0] != kExtent;
  int64_t count = output.sizes[0] < kExtent ? output.sizes[0] : kExtent;
  for (int64_t i = 0; i < count; ++i) {
    int32_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s bin %lld: got %08x, expected %08x\n", label, (long long)i,
              (unsigned)actual, (unsigned)expected[i]);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}

int main(void) {
  int failed = 0;
  uint32_t state = 0xC0FF6401u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    int32_t input[kExtent];
    char label[40];
    for (unsigned i = 0; i < kExtent; ++i) {
      state = nextState(state);
      input[i] = toSigned32(state);
    }
    if (trial == 0)
      for (unsigned i = 0; i < kExtent; ++i)
        input[i] = pack(
            (struct Complex){(i & 1) ? INT16_MIN : INT16_MAX, (i & 2) ? INT16_MAX : INT16_MIN});
    snprintf(label, sizeof label, "forward trial %d", trial);
    failed |= check(_mlir_ciface_cfft64_forward_q15, kForwardTwiddles64, input, label);
    snprintf(label, sizeof label, "inverse trial %d", trial);
    failed |= check(_mlir_ciface_cfft64_inverse_q15, kInverseTwiddles64, input, label);
  }

  /* Directed corpus: a full-scale real or imaginary impulse at EVERY
   * position walks energy through every stage/twiddle index pair of both
   * recursions (signs alternate to exercise both saturating directions),
   * plus complex DC rails. */
  int32_t directed[kExtent];
  for (unsigned position = 0; position < kExtent; ++position) {
    char label[48];
    for (unsigned i = 0; i < kExtent; ++i)
      directed[i] = 0;
    directed[position] = pack((struct Complex){(position & 1) ? INT16_MIN : INT16_MAX, 0});
    snprintf(label, sizeof label, "real impulse %u forward", position);
    failed |= check(_mlir_ciface_cfft64_forward_q15, kForwardTwiddles64, directed, label);
    snprintf(label, sizeof label, "real impulse %u inverse", position);
    failed |= check(_mlir_ciface_cfft64_inverse_q15, kInverseTwiddles64, directed, label);
    directed[position] = pack((struct Complex){0, (position & 1) ? INT16_MAX : INT16_MIN});
    snprintf(label, sizeof label, "imag impulse %u forward", position);
    failed |= check(_mlir_ciface_cfft64_forward_q15, kForwardTwiddles64, directed, label);
    snprintf(label, sizeof label, "imag impulse %u inverse", position);
    failed |= check(_mlir_ciface_cfft64_inverse_q15, kInverseTwiddles64, directed, label);
  }
  for (unsigned i = 0; i < kExtent; ++i)
    directed[i] = pack((struct Complex){INT16_MAX, INT16_MIN});
  failed |= check(_mlir_ciface_cfft64_forward_q15, kForwardTwiddles64, directed, "dc forward");
  failed |= check(_mlir_ciface_cfft64_inverse_q15, kInverseTwiddles64, directed, "dc inverse");
  return failed;
}
