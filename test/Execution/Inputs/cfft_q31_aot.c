/* Independent reference for the packed-Q31 radix-2 CFFT contract.
 *
 * Every intermediate here is __int128 because the contract says the products
 * are exact, not because this corpus can wrap an int64_t. Inside the CFFT the
 * twiddles are frozen unit-circle constants, so both cross sums are bounded
 * by 2^31 * max(|wr| + |wi|) = 0.7071 * INT64_MAX and no input reaches an i64
 * overflow. The carrier width is witnessed at the operation level instead,
 * where ondsp.cx_butterfly admits an arbitrary twiddle: see
 * test/Execution/Inputs/cx_butterfly_q31_aot.c. What this reference does
 * establish is the recursion, the packing, the staged scaling, and the
 * saturating requantization, all written from the contract rather than
 * shared with the compiler. */

#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI64;

extern void _mlir_ciface_cfft4_forward_q31(MemRefI64 *, MemRefI64 *);
extern void _mlir_ciface_cfft4_inverse_q31(MemRefI64 *, MemRefI64 *);
extern void _mlir_ciface_cfft8_forward_q31(MemRefI64 *, MemRefI64 *);
extern void _mlir_ciface_cfft8_inverse_q31(MemRefI64 *, MemRefI64 *);
extern void _mlir_ciface_cfft16_forward_q31(MemRefI64 *, MemRefI64 *);
extern void _mlir_ciface_cfft16_inverse_q31(MemRefI64 *, MemRefI64 *);
extern void _mlir_ciface_cfft8_round_trip_q31(MemRefI64 *, MemRefI64 *);

enum { kMaxExtent = 16, kTrialCount = 16 };

struct Complex {
  int32_t real;
  int32_t imaginary;
};

/* Twiddles w_16^k = cos(2*pi*k/16) -+ j*sin(2*pi*k/16), quantized to Q31 with
 * round-half-even and declared saturation at +1.0, derived independently with
 * 50-digit mpmath from the contract equation. A stage of size n uses
 * table[k * (16 / n)]. The inverse table is listed rather than negated: at
 * k = 4 the forward imaginary part is exactly -2147483648 while its inverse
 * counterpart saturates to 2147483647, so the two are not related by a sign
 * flip. */
static const struct Complex kForwardTwiddles16[8] = {{2147483647, 0},
                                                     {1984016189, -821806413},
                                                     {1518500250, -1518500250},
                                                     {821806413, -1984016189},
                                                     {0, -2147483648},
                                                     {-821806413, -1984016189},
                                                     {-1518500250, -1518500250},
                                                     {-1984016189, -821806413}};
static const struct Complex kInverseTwiddles16[8] = {
    {2147483647, 0}, {1984016189, 821806413},  {1518500250, 1518500250},  {821806413, 1984016189},
    {0, 2147483647}, {-821806413, 1984016189}, {-1518500250, 1518500250}, {-1984016189, 821806413}};

/* Nearest-even signed right shift with saturation into i32, evaluated on the
 * exact __int128 value. Floor division plus a remainder test is total: the
 * add-half-then-shift shortcut would overflow exactly at the saturating
 * boundaries this gate exists to exercise. */
static int32_t requantize(__int128 value, unsigned shift) {
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
  if (quotient < INT32_MIN)
    return INT32_MIN;
  if (quotient > INT32_MAX)
    return INT32_MAX;
  return (int32_t)quotient;
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

static void butterfly(int64_t aBits, int64_t bBits, struct Complex w, int64_t *out0,
                      int64_t *out1) {
  struct Complex a = unpack(aBits);
  struct Complex b = unpack(bBits);
  int32_t tr = requantize((__int128)b.real * w.real - (__int128)b.imaginary * w.imaginary, 31);
  int32_t ti = requantize((__int128)b.real * w.imaginary + (__int128)b.imaginary * w.real, 31);
  *out0 = pack((struct Complex){
      requantize((__int128)a.real + tr, 1),
      requantize((__int128)a.imaginary + ti, 1),
  });
  *out1 = pack((struct Complex){
      requantize((__int128)a.real - tr, 1),
      requantize((__int128)a.imaginary - ti, 1),
  });
}

static void cfftRecursive(const int64_t *input, unsigned stride, unsigned offset, unsigned n,
                          const struct Complex *twiddles, int64_t *output) {
  if (n == 1) {
    output[0] = input[offset];
    return;
  }
  int64_t even[kMaxExtent / 2];
  int64_t odd[kMaxExtent / 2];
  cfftRecursive(input, stride * 2, offset, n / 2, twiddles, even);
  cfftRecursive(input, stride * 2, offset + stride, n / 2, twiddles, odd);
  for (unsigned k = 0; k < n / 2; ++k)
    butterfly(even[k], odd[k], twiddles[k * (kMaxExtent / n)], &output[k], &output[k + n / 2]);
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int compare(const char *label, unsigned extent, const MemRefI64 *output,
                   const int64_t *expected) {
  int failed = output->sizes[0] != (int64_t)extent;
  int64_t count = output->sizes[0] < (int64_t)extent ? output->sizes[0] : (int64_t)extent;
  for (int64_t i = 0; i < count; ++i) {
    int64_t actual = output->aligned[output->offset + i * output->strides[0]];
    if (actual != expected[i]) {
      fprintf(stderr, "%s bin %" PRId64 ": got %016" PRIx64 ", expected %016" PRIx64 "\n", label, i,
              (uint64_t)actual, (uint64_t)expected[i]);
      failed = 1;
    }
  }
  return failed;
}

static int check(void (*kernel)(MemRefI64 *, MemRefI64 *), unsigned extent, bool inverse,
                 const int64_t *input, const char *label) {
  MemRefI64 inputRef = {(int64_t *)input, (int64_t *)input, 0, {(int64_t)extent}, {1}};
  MemRefI64 output;
  kernel(&output, &inputRef);

  int64_t expected[kMaxExtent];
  cfftRecursive(input, 1, 0, extent, inverse ? kInverseTwiddles16 : kForwardTwiddles16, expected);
  int failed = compare(label, extent, &output, expected);
  free(output.allocated);
  return failed;
}

/* Directed rails: the packed full-scale corners, which drive both declared
 * saturations (the product requantization and the per-stage output scale). */
static void fillDirected(unsigned index, unsigned extent, int64_t *input) {
  static const struct Complex kRails[8] = {{INT32_MAX, INT32_MAX}, {INT32_MIN, INT32_MIN},
                                           {INT32_MAX, INT32_MIN}, {INT32_MIN, INT32_MAX},
                                           {INT32_MAX, 0},         {INT32_MIN, 0},
                                           {0, INT32_MAX},         {0, INT32_MIN}};
  for (unsigned i = 0; i < extent; ++i) {
    switch (index) {
    case 0:
      input[i] = pack((struct Complex){0, 0});
      break;
    case 1:
      input[i] = pack(kRails[0]);
      break;
    case 2:
      input[i] = pack(kRails[1]);
      break;
    case 3: /* alternating full-scale rails maximize every stage sum. */
      input[i] = pack(kRails[i % 2]);
      break;
    case 4:
      input[i] = pack(kRails[i % 8]);
      break;
    case 5: /* single full-scale impulse: flat spectrum at the rail. */
      input[i] = i == 0 ? pack(kRails[0]) : pack((struct Complex){0, 0});
      break;
    case 6: /* smallest magnitudes exercise the rounding decisions. */
      input[i] = pack((struct Complex){(int32_t)(i + 1), -(int32_t)(2 * i + 1)});
      break;
    default:
      input[i] = pack((struct Complex){INT32_MIN, INT32_MAX});
      break;
    }
  }
}

enum { kDirectedCount = 8 };

static int checkExtent(void (*forward)(MemRefI64 *, MemRefI64 *),
                       void (*inverse)(MemRefI64 *, MemRefI64 *), unsigned extent,
                       const char *name) {
  int failed = 0;
  int64_t input[kMaxExtent];
  char label[64];

  for (unsigned directed = 0; directed < kDirectedCount; ++directed) {
    fillDirected(directed, extent, input);
    snprintf(label, sizeof(label), "%s directed %u forward", name, directed);
    failed |= check(forward, extent, false, input, label);
    snprintf(label, sizeof(label), "%s directed %u inverse", name, directed);
    failed |= check(inverse, extent, true, input, label);
  }

  uint32_t state = 0x1f2e3d4cu + extent;
  for (unsigned trial = 0; trial < kTrialCount; ++trial) {
    for (unsigned i = 0; i < extent; ++i) {
      state = nextState(state);
      uint32_t real = state;
      state = nextState(state);
      input[i] = pack((struct Complex){decodeSigned32(real), decodeSigned32(state)});
    }
    snprintf(label, sizeof(label), "%s trial %u forward", name, trial);
    failed |= check(forward, extent, false, input, label);
    snprintf(label, sizeof(label), "%s trial %u inverse", name, trial);
    failed |= check(inverse, extent, true, input, label);
  }
  return failed;
}

/* Forward then inverse over the same contract. Each of the 2*log2(N) stages
 * applies one declared output shift of 1, so the staged chain divides by N
 * twice while the unnormalized DFT/IDFT pair multiplies by N once: the
 * composition carries a documented 1/N scaling. It is NOT an identity and
 * must not be "corrected" anywhere in the pipeline. The gate checks bit-exact
 * agreement with the reference composition, and separately that a well-scaled
 * input comes back at 1/N within the accumulated requantization error. */
static int checkRoundTrip(void) {
  enum { kExtent = 8 };
  int failed = 0;
  int64_t input[kExtent];
  uint32_t state = 0x9e3779b9u;

  for (unsigned trial = 0; trial < kTrialCount; ++trial) {
    for (unsigned i = 0; i < kExtent; ++i) {
      state = nextState(state);
      /* Keep the trial inputs inside +-2^28 so the analytic scaling check
       * below is meaningful; the saturating corners are covered by the
       * directed rails of the forward and inverse gates. */
      int32_t real = (int32_t)(state >> 4) - (int32_t)0x08000000;
      state = nextState(state);
      int32_t imaginary = (int32_t)(state >> 4) - (int32_t)0x08000000;
      input[i] = pack((struct Complex){real, imaginary});
    }

    MemRefI64 inputRef = {input, input, 0, {kExtent}, {1}};
    MemRefI64 output;
    _mlir_ciface_cfft8_round_trip_q31(&output, &inputRef);

    int64_t spectrum[kExtent];
    int64_t expected[kExtent];
    cfftRecursive(input, 1, 0, kExtent, kForwardTwiddles16, spectrum);
    cfftRecursive(spectrum, 1, 0, kExtent, kInverseTwiddles16, expected);
    failed |= compare("round trip", kExtent, &output, expected);

    /* Each element passes the full staged boundary budget of both
     * transforms: 2*log2(8) = 6 output scales, plus up to that many product
     * requantizations along the b-side of each butterfly. The +-6 LSB window
     * below is the empirical bound this gate enforces, not a derived error
     * bound. */
    for (unsigned i = 0; i < kExtent; ++i) {
      struct Complex source = unpack(input[i]);
      struct Complex result = unpack(expected[i]);
      int64_t realError = (int64_t)result.real - (int64_t)source.real / (int64_t)kExtent;
      int64_t imaginaryError =
          (int64_t)result.imaginary - (int64_t)source.imaginary / (int64_t)kExtent;
      if (realError > 6 || realError < -6 || imaginaryError > 6 || imaginaryError < -6) {
        fprintf(stderr,
                "round trip scaling %u: (%" PRId32 ", %" PRId32 ") is not input/%d of "
                "(%" PRId32 ", %" PRId32 ")\n",
                i, result.real, result.imaginary, kExtent, source.real, source.imaginary);
        failed = 1;
      }
    }
    free(output.allocated);
  }
  return failed;
}

int main(void) {
  int failed = 0;
  failed |= checkExtent(_mlir_ciface_cfft4_forward_q31, _mlir_ciface_cfft4_inverse_q31, 4, "N=4");
  failed |= checkExtent(_mlir_ciface_cfft8_forward_q31, _mlir_ciface_cfft8_inverse_q31, 8, "N=8");
  failed |=
      checkExtent(_mlir_ciface_cfft16_forward_q31, _mlir_ciface_cfft16_inverse_q31, 16, "N=16");
  failed |= checkRoundTrip();
  return failed;
}
