#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for vertical (cross-output) batching of the valid-boundary f32
 * filter.
 *
 * The batched lanes carry independent outputs (order preserving; the pass
 * description carries the argument), so this harness checks the falsifiable
 * consequence: batched == ordered == an independent reference, per element.
 *
 * Both admitted contracts are EXACT: off names one rounded product followed by
 * one accumulation per tap, fma names one fused event per tap. Neither permits
 * a relation, so this harness compares bits rather than magnitudes. Each
 * reference below is written directly from its contract and shares nothing
 * with the compiler, and the object is built with -ffp-contract=off so the C
 * compiler cannot quietly turn the off reference into the fma one.
 *
 * Every corpus entry is checked four ways per output: batched off against the
 * off reference, batched fma against the fma reference, and each batched
 * kernel against the dynamically shaped kernel in the same object. The dynamic
 * kernels are the ordered oracle — the pass refuses a loop whose extents it
 * cannot see, so they carry the schedule the batching had to preserve, without
 * a second compilation.
 *
 * The directed families are chosen for the properties a lane-parallel body can
 * silently lose: NaN and infinity payload propagation across lanes and taps,
 * the sign of a zero sum, subnormal products and sums that a flushing lane
 * would zero, and one cancellation pair whose exported bits differ between the
 * two contracts. The last family is required to actually separate off from
 * fma, so the pair of profiles is not carried vacuously. */

enum {
  kInputLength = 40,
  kTapCount = 8,
  kOutputLength = 33,
  kVectorWidth = 8,
};

/* kOutputLength is not a multiple of kVectorWidth, so the batched loop covers
 * outputs 0..31 and output 32 stays on the untouched ordered loop. Both code
 * paths therefore execute in every corpus entry. */

/* Non-vacuity obligations a corpus entry can carry. Without them a directed
 * family can degenerate into another finite random trial. */
enum {
  kRequireNothing = 0,
  /* Some output must separate the two contracts in its exported bits. */
  kRequireContractSplit = 1 << 0,
  /* Some output must be NaN, so the NaN-tolerant comparison is exercised
   * rather than merely permitted. */
  kRequireNaN = 1 << 1,
  /* Some output must be a nonzero subnormal, which is what pins that no stage
   * flushed a subnormal product or sum to zero. */
  kRequireSubnormal = 1 << 2,
};

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32;

#define MAKE_MEMREF(DATA, COUNT)                                                                   \
  {                                                                                                \
    DATA, DATA, 0, {COUNT}, { 1 }                                                                  \
  }

/* The bufferized kernels write through the init descriptor and return a
 * descriptor aliasing it, so the caller owns the storage and nothing here is
 * freed. */
typedef void (*Kernel)(MemRefF32 *, MemRefF32 *, MemRefF32 *, MemRefF32 *);

extern void _mlir_ciface_f32_filter_off_batched(MemRefF32 *, MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_filter_fma_batched(MemRefF32 *, MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_filter_off_ordered(MemRefF32 *, MemRefF32 *, MemRefF32 *, MemRefF32 *);
extern void _mlir_ciface_f32_filter_fma_ordered(MemRefF32 *, MemRefF32 *, MemRefF32 *, MemRefF32 *);

/* A value no corpus entry can produce, written into every output before the
 * call so a position the kernel failed to write is caught by the comparison
 * rather than read as a coincidence. */
static const float kSentinel = 0x1.b4b4b4p+53f;

/* The off contract, transcribed: for each tap one product is formed and
 * rounded, then the accumulator observes it in a separate addition. The two
 * statements are deliberately not one expression. */
static void reference_off(const float *input, const float *coefficients, float *output) {
  for (int64_t result = 0; result < kOutputLength; ++result) {
    float accumulator = 0.0f;
    for (int64_t tap = 0; tap < kTapCount; ++tap) {
      const float product = input[result + tap] * coefficients[tap];
      accumulator = accumulator + product;
    }
    output[result] = accumulator;
  }
}

/* The fma contract, transcribed: one fused event per tap, so the product is
 * never rounded on its own. */
static void reference_fma(const float *input, const float *coefficients, float *output) {
  for (int64_t result = 0; result < kOutputLength; ++result) {
    float accumulator = 0.0f;
    for (int64_t tap = 0; tap < kTapCount; ++tap)
      accumulator = fmaf(input[result + tap], coefficients[tap], accumulator);
    output[result] = accumulator;
  }
}

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* Two NaNs count as equal because the contract does not name a payload;
 * everything else, including the sign of a zero, must match bit for bit. */
static int bitEqual(float lhs, float rhs) {
  if (isnan(lhs) && isnan(rhs))
    return 1;
  return floatBits(lhs) == floatBits(rhs);
}

/* Runs one kernel over one corpus entry into a fresh sentinel-filled output,
 * then reads the result back through the returned descriptor. */
static int runKernel(const char *caseName, const char *kernelName, Kernel kernel,
                     const float *input, const float *coefficients, float *output) {
  float inputCopy[kInputLength];
  float coefficientCopy[kTapCount];
  float outputBuffer[kOutputLength];

  memcpy(inputCopy, input, sizeof(inputCopy));
  memcpy(coefficientCopy, coefficients, sizeof(coefficientCopy));
  for (int64_t position = 0; position < kOutputLength; ++position)
    outputBuffer[position] = kSentinel;

  MemRefF32 inputRef = MAKE_MEMREF(inputCopy, kInputLength);
  MemRefF32 coefficientRef = MAKE_MEMREF(coefficientCopy, kTapCount);
  MemRefF32 initRef = MAKE_MEMREF(outputBuffer, kOutputLength);
  MemRefF32 result;
  kernel(&result, &inputRef, &coefficientRef, &initRef);

  if (result.sizes[0] != kOutputLength) {
    fprintf(stderr, "%s/%s: result length %lld, expected %d\n", caseName, kernelName,
            (long long)result.sizes[0], kOutputLength);
    return 1;
  }
  for (int64_t position = 0; position < kOutputLength; ++position)
    output[position] = result.aligned[result.offset + position * result.strides[0]];
  return 0;
}

static int compareSeries(const char *caseName, const char *description, const float *actual,
                         const float *expected) {
  int failed = 0;
  for (int64_t position = 0; position < kOutputLength; ++position) {
    if (bitEqual(actual[position], expected[position]))
      continue;
    fprintf(stderr, "%s/%s[%lld]: got 0x%08x, expected 0x%08x\n", caseName, description,
            (long long)position, floatBits(actual[position]), floatBits(expected[position]));
    failed = 1;
  }
  return failed;
}

/* One corpus entry: both contracts, both schedules, and the independent
 * references, compared per output. `required` states what the entry must
 * actually exhibit for its family to mean anything. */
static int checkCase(const char *name, const float *input, const float *coefficients,
                     unsigned required) {
  float expectedOff[kOutputLength];
  float expectedFma[kOutputLength];
  float batchedOff[kOutputLength];
  float batchedFma[kOutputLength];
  float orderedOff[kOutputLength];
  float orderedFma[kOutputLength];

  reference_off(input, coefficients, expectedOff);
  reference_fma(input, coefficients, expectedFma);

  int failed = 0;
  failed |= runKernel(name, "batched off", _mlir_ciface_f32_filter_off_batched, input, coefficients,
                      batchedOff);
  failed |= runKernel(name, "batched fma", _mlir_ciface_f32_filter_fma_batched, input, coefficients,
                      batchedFma);
  failed |= runKernel(name, "ordered off", _mlir_ciface_f32_filter_off_ordered, input, coefficients,
                      orderedOff);
  failed |= runKernel(name, "ordered fma", _mlir_ciface_f32_filter_fma_ordered, input, coefficients,
                      orderedFma);
  if (failed)
    return failed;

  failed |= compareSeries(name, "batched off vs reference off", batchedOff, expectedOff);
  failed |= compareSeries(name, "batched fma vs reference fma", batchedFma, expectedFma);
  failed |= compareSeries(name, "batched off vs ordered off", batchedOff, orderedOff);
  failed |= compareSeries(name, "batched fma vs ordered fma", batchedFma, orderedFma);

  if ((required & kRequireContractSplit) != 0) {
    int split = 0;
    for (int64_t position = 0; position < kOutputLength; ++position)
      if (!bitEqual(expectedOff[position], expectedFma[position]))
        split = 1;
    if (!split) {
      fprintf(stderr,
              "%s: the two contracts agree at every output, so the entry cannot tell a "
              "fused update from a separated one\n",
              name);
      failed = 1;
    }
  }
  if ((required & kRequireNaN) != 0) {
    int sawNaN = 0;
    for (int64_t position = 0; position < kOutputLength; ++position)
      if (isnan(expectedOff[position]) && isnan(expectedFma[position]))
        sawNaN = 1;
    if (!sawNaN) {
      fprintf(stderr, "%s: no output is NaN, so the entry exercises no NaN propagation\n", name);
      failed = 1;
    }
  }
  if ((required & kRequireSubnormal) != 0) {
    int sawSubnormal = 0;
    for (int64_t position = 0; position < kOutputLength; ++position)
      if (fpclassify(expectedOff[position]) == FP_SUBNORMAL &&
          fpclassify(expectedFma[position]) == FP_SUBNORMAL)
        sawSubnormal = 1;
    if (!sawSubnormal) {
      fprintf(stderr,
              "%s: no output is a nonzero subnormal, so the entry says nothing about "
              "flushing\n",
              name);
      failed = 1;
    }
  }
  return failed;
}

static uint32_t nextRandom(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

/* Draws a finite float by construction rather than by filtering: the sign and
 * the mantissa come from one xorshift word and the biased exponent from a
 * window spanning 2^-17 to 2^17, so eight products still sum well inside the
 * finite range while the operands span enough binades for the accumulation
 * order to be observable. Specials are directed, never drawn. */
static float randomFinite(uint32_t *state) {
  const uint32_t draw = nextRandom(state);
  const uint32_t exponent = 110 + nextRandom(state) % 35;
  const uint32_t bits =
      (draw & UINT32_C(0x80000000)) | (exponent << 23) | (draw & UINT32_C(0x007fffff));
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static int checkRandomCorpus(void) {
  float input[kInputLength];
  float coefficients[kTapCount];
  uint32_t state = UINT32_C(0x9e3779b9);
  int failed = 0;

  for (int trial = 0; trial < 16; ++trial) {
    char name[32];
    snprintf(name, sizeof(name), "random %d", trial);
    for (int64_t position = 0; position < kInputLength; ++position)
      input[position] = randomFinite(&state);
    for (int64_t tap = 0; tap < kTapCount; ++tap)
      coefficients[tap] = randomFinite(&state);
    failed |= checkCase(name, input, coefficients, kRequireNothing);
  }
  return failed;
}

/* A finite background the directed families plant their specials into. The
 * values span several binades so a special never lands in an otherwise
 * uniform window. */
static void fillBackground(float *input, float *coefficients) {
  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = (position % 2 == 0 ? 1.0f : -1.0f) * (0.5f + (float)(position % 5)) *
                      (float)(1 << (position % 7));
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = (tap % 3 == 0 ? -1.0f : 1.0f) * (0.25f + 0.5f * (float)tap);
}

/* (a) One NaN per entry, moved across the input so it reaches a different lane
 * of a different batched block and a different tap of the affected windows.
 * Position 39 is reached only by output 32, which is the ordered remainder. */
static int checkNaNCorpus(void) {
  static const int64_t positions[] = {0, 3, 9, 17, 26, 39};
  float input[kInputLength];
  float coefficients[kTapCount];
  int failed = 0;

  for (size_t index = 0; index < sizeof(positions) / sizeof(positions[0]); ++index) {
    char name[32];
    snprintf(name, sizeof(name), "nan at input %lld", (long long)positions[index]);
    fillBackground(input, coefficients);
    input[positions[index]] = NAN;
    failed |= checkCase(name, input, coefficients, kRequireNaN);
  }
  return failed;
}

/* (b) Infinities. Adjacent opposite infinities let a window that covers both
 * produce an Inf + -Inf NaN while its neighbours stay infinite, and an
 * infinity meeting a zero coefficient produces the NaN in the product instead
 * of in the sum. */
static int checkInfinityCorpus(void) {
  float input[kInputLength];
  float coefficients[kTapCount];
  int failed = 0;

  fillBackground(input, coefficients);
  input[12] = INFINITY;
  input[13] = -INFINITY;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = 1.0f;
  failed |= checkCase("adjacent infinities", input, coefficients, kRequireNaN);

  fillBackground(input, coefficients);
  input[20] = -INFINITY;
  coefficients[4] = 0.0f;
  failed |= checkCase("infinity against a zero coefficient", input, coefficients, kRequireNaN);
  return failed;
}

/* (c) Signed zeros. Every product is a zero here, so the exported sign is
 * decided entirely by the accumulator's initial value: starting a lane from
 * -0.0 instead of +0.0 turns every one of these outputs negative, and starting
 * the remainder differently from the batched block splits the two. */
static int checkSignedZeroCorpus(void) {
  float input[kInputLength];
  float coefficients[kTapCount];
  int failed = 0;

  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = -0.0f;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = 0.0f;
  failed |=
      checkCase("negative zeros against positive zeros", input, coefficients, kRequireNothing);

  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = (position % 2 == 0) ? -0.0f : 0.0f;
  coefficients[0] = 0.0f;
  coefficients[1] = -0.0f;
  coefficients[2] = -1.0f;
  coefficients[3] = -0.5f;
  coefficients[4] = 0.0f;
  coefficients[5] = -2.0f;
  coefficients[6] = -0.0f;
  coefficients[7] = 3.0f;
  failed |= checkCase("mixed zero signs", input, coefficients, kRequireNothing);
  return failed;
}

/* (d) Subnormals. The inputs sit at 2^-140 magnitudes and the coefficients are
 * small powers of two, so every product and every partial sum stays subnormal.
 * A stage that flushed either to zero would change the exported bits, and the
 * entry refuses to pass unless some output really is a nonzero subnormal. */
static int checkSubnormalCorpus(void) {
  static const float scales[kTapCount] = {0x1p-5f, -0x1p-6f, 0x1p-7f, -0x1p-8f,
                                          0x1p-9f, -0x1p-4f, 0x1p-3f, -0x1p-2f};
  float input[kInputLength];
  float coefficients[kTapCount];

  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] =
        (position % 2 == 0 ? 1.0f : -1.0f) * (1.0f + 0.25f * (float)(position % 7)) * 0x1p-140f;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = scales[tap];
  return checkCase("subnormal products and sums", input, coefficients, kRequireSubnormal);
}

/* (e) The cancellation pair that separates the two contracts. At an even
 * output the first tap leaves the accumulator at exactly 1.0 and the second
 * tap's exact product is just above 2^-24: rounding it first lands on the
 * 2^-24 tie, which the following addition resolves back down to 1.0, while the
 * fused update sees the excess and rounds up to 1 + 2^-23. The remaining taps
 * have zero coefficients, so that one disagreement is what reaches the
 * exported bits — including at output 32, on the ordered remainder. */
static int checkContractSplitCorpus(void) {
  float input[kInputLength];
  float coefficients[kTapCount];

  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = (position % 2 == 0) ? 1.0f : 0x1.000002p+0f;
  coefficients[0] = 1.0f;
  coefficients[1] = 0x1.fffffep-25f;
  for (int64_t tap = 2; tap < kTapCount; ++tap)
    coefficients[tap] = 0.0f;
  return checkCase("cancellation pair", input, coefficients, kRequireContractSplit);
}

int main(void) {
  int failed = 0;
  failed |= checkRandomCorpus();
  failed |= checkNaNCorpus();
  failed |= checkInfinityCorpus();
  failed |= checkSignedZeroCorpus();
  failed |= checkSubnormalCorpus();
  failed |= checkContractSplitCorpus();
  return failed;
}
