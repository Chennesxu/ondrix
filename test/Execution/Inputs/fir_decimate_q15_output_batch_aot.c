#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Object gate for vertical (cross-output) batching of Q15 decimation.
 *
 * The batched lanes carry independent outputs (order preserving; the pass
 * description carries the argument). This harness pins that claim where it is
 * falsifiable: batched == ordered == an independent reference, per element,
 * for both executable accumulator profiles.
 *
 * Two tap counts, because an accumulator rail is only evidence when the clamp
 * survives to the exported bits.
 *
 * Eight taps reach the i34 rail exactly: eight Q15 full products of
 * -32768 * -32768 = 2^30 sum to 2^33, one past the i34 maximum, so a saturating
 * i34 accumulator clamps on its last tap. That exercises the clamp inside the
 * vector body, but it is all eight taps can do — the clamped accumulator is far
 * outside the Q15 destination and the export saturates, which would hide an
 * implementation that clamped too much.
 *
 * Seventeen taps make the clamp OBSERVABLE. Nine rail taps drive the value past
 * 2^33 twice, the saturating trajectory pins at 2^33 - 1, and eight
 * opposite-sign taps pull it down to 262143, exported as 8 — inside the
 * destination. The same corpus on the i40 profile never clamps and exports the
 * saturated 32767. `expectVisibleClamp` requires exactly that disagreement at
 * an i34 output that is itself non-saturating, so dropping the clamp, or
 * applying one lane's clamp to another lane, is caught in the exported bits
 * rather than swallowed by the destination range.
 *
 * This is the first committed strongly visible construction, not a proven
 * minimum: it separates the two trajectories by 2^30, and shorter shapes can
 * still be made visible through the one-raw-unit clamp loss landing on a
 * nearest-even half-tie. A tighter shape would be a narrower witness of the
 * same property, so it is not committed here. */

enum {
  kOutputLength = 19,
  kFactor = 2,
  kVectorWidth = 8,
  kShortTaps = 8,
  kShortInput = 44,
  kLongTaps = 17,
  kLongInput = 53,
  kMaxTaps = 17,
  kMaxInput = 53,
};

/* kOutputLength is odd, so with kVectorWidth = 8 the batched loop covers two
 * full blocks (outputs 0..15) and outputs 16, 17, 18 stay on the untouched
 * ordered loop. Both code paths therefore execute in every trial. */

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

typedef void (*Kernel)(MemRefI16 *, MemRefI16 *, MemRefI16 *, MemRefI16 *);

extern void _mlir_ciface_decimate_short_i40_batched(MemRefI16 *, MemRefI16 *, MemRefI16 *,
                                                    MemRefI16 *);
extern void _mlir_ciface_decimate_short_i34_batched(MemRefI16 *, MemRefI16 *, MemRefI16 *,
                                                    MemRefI16 *);
extern void _mlir_ciface_decimate_long_i40_batched(MemRefI16 *, MemRefI16 *, MemRefI16 *,
                                                   MemRefI16 *);
extern void _mlir_ciface_decimate_long_i34_batched(MemRefI16 *, MemRefI16 *, MemRefI16 *,
                                                   MemRefI16 *);
extern void _mlir_ciface_decimate_i40_ordered(MemRefI16 *, MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_decimate_i34_ordered(MemRefI16 *, MemRefI16 *, MemRefI16 *, MemRefI16 *);

static const struct Policy policy_i40 = {
    .width = 16,
    .frac = 15,
    .accumulator_width = 40,
    .accumulator_frac = 30,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

static const struct Policy policy_i34 = {
    .width = 16,
    .frac = 15,
    .accumulator_width = 34,
    .accumulator_frac = 30,
    .update_overflow = SATURATE,
    .state_rounding = NEAREST_EVEN,
    .state_overflow = SATURATE,
    .output_rounding = NEAREST_EVEN,
    .output_overflow = SATURATE,
};

/* Independent reference: the declared ordered fold, one output at a time, with
 * no knowledge of lanes. `clamped` counts the updates whose exact sum left the
 * accumulator range, which is what makes the rail claim non-vacuous. */
static void reference(const int16_t *input, const int16_t *coefficients, int64_t tapCount,
                      int16_t *output, const struct Policy *policy, int64_t *clamped) {
  const __int128 minimum = -((__int128)1 << (policy->accumulator_width - 1));
  const __int128 maximum = ((__int128)1 << (policy->accumulator_width - 1)) - 1;
  for (int64_t result = 0; result < kOutputLength; ++result) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < tapCount; ++tap) {
      __int128 exact = (__int128)accumulator +
                       (__int128)input[result * kFactor + tap] * (__int128)coefficients[tap];
      if (exact < minimum || exact > maximum)
        ++*clamped;
      accumulator =
          update_reference(accumulator, input[result * kFactor + tap], coefficients[tap], policy);
    }
    output[result] = (int16_t)export_reference(accumulator, policy->output_rounding,
                                               policy->output_overflow, policy);
  }
}

static void setDescriptor(MemRefI16 *descriptor, int16_t *data, int64_t length) {
  descriptor->allocated = data;
  descriptor->aligned = data;
  descriptor->offset = 0;
  descriptor->sizes[0] = length;
  descriptor->strides[0] = 1;
}

struct Shape {
  const char *name;
  int64_t tapCount;
  int64_t inputLength;
  Kernel i40Batched;
  Kernel i34Batched;
};

/* Runs one corpus through one shape and one accumulator profile: the batched
 * kernel and the in-object ordered oracle, both compared element by element
 * against the independent reference. `exported` receives the batched result so
 * the caller can also compare the two profiles against each other. */
static int checkProfile(const struct Shape *shape, const char *name, const char *profileName,
                        const struct Policy *policy, Kernel batchedKernel, Kernel orderedKernel,
                        const int16_t *input, const int16_t *coefficients, int16_t *exported,
                        int64_t *clamped) {
  int16_t expected[kOutputLength];
  int16_t batched[kOutputLength];
  int16_t ordered[kOutputLength];
  int16_t inputCopy[kMaxInput];
  int16_t coefficientCopy[kMaxTaps];

  *clamped = 0;
  reference(input, coefficients, shape->tapCount, expected, policy, clamped);

  for (int64_t position = 0; position < shape->inputLength; ++position)
    inputCopy[position] = input[position];
  for (int64_t position = 0; position < shape->tapCount; ++position)
    coefficientCopy[position] = coefficients[position];
  for (int64_t position = 0; position < kOutputLength; ++position) {
    /* A sentinel the kernels must overwrite everywhere. */
    batched[position] = 0x5a5a;
    ordered[position] = 0x5a5a;
  }

  MemRefI16 inputRef;
  MemRefI16 coefficientRef;
  MemRefI16 batchedRef;
  MemRefI16 orderedRef;
  MemRefI16 batchedResult;
  MemRefI16 orderedResult;
  setDescriptor(&inputRef, inputCopy, shape->inputLength);
  setDescriptor(&coefficientRef, coefficientCopy, shape->tapCount);
  setDescriptor(&batchedRef, batched, kOutputLength);
  setDescriptor(&orderedRef, ordered, kOutputLength);
  batchedKernel(&batchedResult, &inputRef, &coefficientRef, &batchedRef);
  orderedKernel(&orderedResult, &inputRef, &coefficientRef, &orderedRef);

  int failed = 0;
  if (batchedResult.sizes[0] != kOutputLength || orderedResult.sizes[0] != kOutputLength) {
    fprintf(stderr, "%s/%s/%s: result length %lld and %lld, expected %d\n", shape->name, name,
            profileName, (long long)batchedResult.sizes[0], (long long)orderedResult.sizes[0],
            kOutputLength);
    failed = 1;
  }
  for (int64_t position = 0; position < kOutputLength; ++position) {
    const int16_t batchedValue =
        batchedResult.aligned[batchedResult.offset + position * batchedResult.strides[0]];
    const int16_t orderedValue =
        orderedResult.aligned[orderedResult.offset + position * orderedResult.strides[0]];
    if (batchedValue != expected[position] || orderedValue != expected[position]) {
      fprintf(stderr, "%s/%s/%s[%lld]: reference %d, batched %d, ordered %d\n", shape->name, name,
              profileName, (long long)position, expected[position], batchedValue, orderedValue);
      failed = 1;
    }
    exported[position] = batchedValue;
  }
  return failed;
}

/* One case, both profiles, three-way equality per element.
 *
 * `expectClamp` requires the i34 accumulator to actually reach its rail.
 * `expectVisibleClamp` additionally requires the clamp to change an exported
 * value that is itself inside the destination range: the i34 and i40 profiles
 * must disagree at some position where the i34 output is neither 32767 nor
 * -32768. Without that second requirement a rail case proves only that the
 * clamp ran, not that anything downstream could observe it. */
static int checkCase(const struct Shape *shape, const char *name, const int16_t *input,
                     const int16_t *coefficients, int expectClamp, int expectVisibleClamp) {
  int16_t exported40[kOutputLength];
  int16_t exported34[kOutputLength];
  int64_t clamped40 = 0;
  int64_t clamped34 = 0;

  int failed =
      checkProfile(shape, name, "i40", &policy_i40, shape->i40Batched,
                   _mlir_ciface_decimate_i40_ordered, input, coefficients, exported40, &clamped40);
  failed |=
      checkProfile(shape, name, "i34", &policy_i34, shape->i34Batched,
                   _mlir_ciface_decimate_i34_ordered, input, coefficients, exported34, &clamped34);

  /* The i40 rail is far outside the reachable sum of seventeen Q15 products, so
   * a clamp there would mean the profile is not what it claims. That also makes
   * i40 the unclamped oracle the visibility check below compares against. */
  if (clamped40 != 0) {
    fprintf(stderr, "%s/%s: the i40 accumulator clamped, which this profile cannot reach\n",
            shape->name, name);
    failed = 1;
  }
  if (expectClamp && clamped34 == 0) {
    fprintf(stderr,
            "%s/%s: the i34 accumulator never reached its rail, so the case proves nothing "
            "about the clamp\n",
            shape->name, name);
    failed = 1;
  }
  if (expectVisibleClamp) {
    int visible = 0;
    for (int64_t position = 0; position < kOutputLength; ++position)
      if (exported34[position] != exported40[position] && exported34[position] != 32767 &&
          exported34[position] != -32768)
        visible = 1;
    if (!visible) {
      fprintf(stderr,
              "%s/%s: no non-saturating i34 output differs from i40, so the clamp is masked at "
              "the destination and the case proves nothing\n",
              shape->name, name);
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

/* The corpora that do not depend on the tap count: rails, impulses, zeros, and
 * pseudo-random trials. */
static int checkSharedCorpus(const struct Shape *shape) {
  int16_t input[kMaxInput];
  int16_t coefficients[kMaxTaps];
  const int64_t taps = shape->tapCount;
  const int64_t length = shape->inputLength;
  int failed = 0;

  /* (a) Every sample and every coefficient at the negative rail. Each product
   * is -32768 * -32768 = 2^30, so the i34 accumulator passes 2^33 and clamps. */
  for (int64_t position = 0; position < length; ++position)
    input[position] = INT16_MIN;
  for (int64_t tap = 0; tap < taps; ++tap)
    coefficients[tap] = INT16_MIN;
  failed |= checkCase(shape, "all-rail", input, coefficients, /*expectClamp=*/1,
                      /*expectVisibleClamp=*/0);

  /* (b) The same rails, but two zero samples every ten positions. A window of
   * rails still clamps while a window covering a zero clamps later or not at
   * all, so the clamp now differs BETWEEN lanes of one batched block. */
  for (int64_t position = 0; position < length; ++position)
    input[position] = (position % 10 >= 8) ? 0 : INT16_MIN;
  failed |= checkCase(shape, "per-lane rail", input, coefficients, /*expectClamp=*/1,
                      /*expectVisibleClamp=*/0);

  /* (c) Alternating sample rails against alternating coefficient rails, so the
   * running value climbs and falls within one window. */
  for (int64_t position = 0; position < length; ++position)
    input[position] = (position % 2 == 0) ? INT16_MIN : INT16_MAX;
  for (int64_t tap = 0; tap < taps; ++tap)
    coefficients[tap] = (tap % 2 == 0) ? INT16_MIN : INT16_MAX;
  failed |= checkCase(shape, "alternating rails", input, coefficients, /*expectClamp=*/0,
                      /*expectVisibleClamp=*/0);

  /* (d) A single impulse inside the batched range with a ramp of coefficients.
   * Exactly one tap of a few windows is nonzero, which catches a shuffle that
   * picks the wrong lane or the wrong phase. */
  for (int64_t position = 0; position < length; ++position)
    input[position] = 0;
  input[17] = INT16_MIN;
  for (int64_t tap = 0; tap < taps; ++tap)
    coefficients[tap] = (int16_t)(1 << (tap % 9 + 6));
  failed |= checkCase(shape, "single impulse", input, coefficients, /*expectClamp=*/0,
                      /*expectVisibleClamp=*/0);

  /* (e) A second impulse near the end, so the ordered remainder is not
   * exercised only by zeros. */
  for (int64_t position = 0; position < length; ++position)
    input[position] = 0;
  input[length - 5] = INT16_MAX;
  failed |= checkCase(shape, "tail impulse", input, coefficients, /*expectClamp=*/0,
                      /*expectVisibleClamp=*/0);

  /* (f) All zeros. The exported value is zero everywhere, which pins that the
   * batched loop wrote every element it claimed. */
  for (int64_t position = 0; position < length; ++position)
    input[position] = 0;
  for (int64_t tap = 0; tap < taps; ++tap)
    coefficients[tap] = INT16_MIN;
  failed |= checkCase(shape, "zeros", input, coefficients, /*expectClamp=*/0,
                      /*expectVisibleClamp=*/0);

  /* (g) Deterministic xorshift32 trials. The first half draws from the full i16
   * range so clamps and export saturation both occur; the second half is
   * bounded so the exported values stay inside the destination and remain
   * rounding sensitive. */
  uint32_t state = UINT32_C(0x9e3779b9);
  for (int trial = 0; trial < 16; ++trial) {
    const int wide = trial < 8;
    for (int64_t position = 0; position < length; ++position) {
      const uint32_t draw = nextRandom(&state);
      input[position] =
          wide ? (int16_t)(draw >> 16) : (int16_t)((int32_t)((draw >> 16) % 4001) - 2000);
    }
    for (int64_t tap = 0; tap < taps; ++tap) {
      const uint32_t draw = nextRandom(&state);
      coefficients[tap] =
          wide ? (int16_t)(draw >> 16) : (int16_t)((int32_t)((draw >> 16) % 4001) - 2000);
    }
    failed |= checkCase(shape, wide ? "random wide" : "random bounded", input, coefficients,
                        /*expectClamp=*/0, /*expectVisibleClamp=*/0);
  }

  return failed;
}

/* The seventeen-tap corpus whose clamp is destructive and visible.
 *
 * Coefficients: nine at -32768, then eight at 32767. With every sample at
 * -32768 the first nine products are +2^30 and the last eight are -1073709056.
 * The saturating i34 trajectory pins at 2^33 - 1 on the eighth tap, is pinned
 * again on the ninth, and the eight negative taps leave 262143, which exports
 * as 8. The i40 trajectory never clamps, ends at 1074003968, and exports the
 * saturated 32767. */
static int checkVisibleClampCorpus(const struct Shape *shape) {
  int16_t input[kMaxInput];
  int16_t coefficients[kMaxTaps];
  int failed = 0;

  for (int64_t tap = 0; tap < kLongTaps; ++tap)
    coefficients[tap] = tap < 9 ? INT16_MIN : INT16_MAX;

  for (int64_t position = 0; position < kLongInput; ++position)
    input[position] = INT16_MIN;
  failed |= checkCase(shape, "rail then return", input, coefficients, /*expectClamp=*/1,
                      /*expectVisibleClamp=*/1);

  /* The same construction with four zeroed samples, so within one batched block
   * some lanes clamp on the rail taps and others clamp later or not at all. The
   * exported values then differ lane by lane, which no uniform per-block clamp
   * could reproduce. */
  input[20] = 0;
  input[21] = 0;
  input[34] = 0;
  input[35] = 0;
  failed |= checkCase(shape, "per-lane rail then return", input, coefficients, /*expectClamp=*/1,
                      /*expectVisibleClamp=*/1);

  return failed;
}

int main(void) {
  static const struct Shape shortShape = {
      "K8",
      kShortTaps,
      kShortInput,
      _mlir_ciface_decimate_short_i40_batched,
      _mlir_ciface_decimate_short_i34_batched,
  };
  static const struct Shape longShape = {
      "K17",
      kLongTaps,
      kLongInput,
      _mlir_ciface_decimate_long_i40_batched,
      _mlir_ciface_decimate_long_i34_batched,
  };

  int failed = 0;
  failed |= checkSharedCorpus(&shortShape);
  failed |= checkSharedCorpus(&longShape);
  failed |= checkVisibleClampCorpus(&longShape);
  return failed;
}
