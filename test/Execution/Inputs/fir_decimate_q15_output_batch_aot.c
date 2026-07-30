#include "fixed_point_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Object gate for vertical (cross-output) batching of Q15 decimation.
 *
 * Batching outputs across accumulator lanes is order preserving: lane j is
 * output m + j and folds the same taps in the same increasing order into its
 * own accumulator. No lane is ever combined with another, so legality does not
 * depend on a range or overflow proof — unlike the horizontal reduction
 * schedules, whose lanes do change the fold order. This harness pins that
 * claim where it is falsifiable: batched == ordered == an independent
 * reference, per element, for both executable accumulator profiles.
 *
 * The i34 profile is the discriminating one. Eight Q15 full products of
 * -32768 * -32768 = 2^30 sum to exactly 2^33, one past the i34 maximum
 * 2^33 - 1, so a saturating i34 accumulator clamps at the last tap. That clamp
 * is destructive and per lane: a window of eight rails clamps while an
 * overlapping window containing one non-rail sample does not. An
 * implementation that saturated a batched partial sum, or that clamped
 * uniformly across lanes, disagrees here. The i40 profile carries the same
 * corpus without ever reaching its own rail, so it isolates the batching from
 * the clamp. */

enum {
  kInputLength = 44,
  kTapCount = 8,
  kOutputLength = 19,
  kFactor = 2,
  kVectorWidth = 8,
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

extern void _mlir_ciface_decimate_i40_batched(MemRefI16 *result, MemRefI16 *input,
                                              MemRefI16 *coefficients, MemRefI16 *init);
extern void _mlir_ciface_decimate_i40_ordered(MemRefI16 *result, MemRefI16 *input,
                                              MemRefI16 *coefficients, MemRefI16 *init);
extern void _mlir_ciface_decimate_i34_batched(MemRefI16 *result, MemRefI16 *input,
                                              MemRefI16 *coefficients, MemRefI16 *init);
extern void _mlir_ciface_decimate_i34_ordered(MemRefI16 *result, MemRefI16 *input,
                                              MemRefI16 *coefficients, MemRefI16 *init);

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
static void reference(const int16_t *input, const int16_t *coefficients, int16_t *output,
                      const struct Policy *policy, int64_t *clamped) {
  const __int128 minimum = -((__int128)1 << (policy->accumulator_width - 1));
  const __int128 maximum = ((__int128)1 << (policy->accumulator_width - 1)) - 1;
  for (int64_t result = 0; result < kOutputLength; ++result) {
    int64_t accumulator = 0;
    for (int64_t tap = 0; tap < kTapCount; ++tap) {
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

struct Profile {
  const char *name;
  const struct Policy *policy;
  void (*batched)(MemRefI16 *, MemRefI16 *, MemRefI16 *, MemRefI16 *);
  void (*ordered)(MemRefI16 *, MemRefI16 *, MemRefI16 *, MemRefI16 *);
};

/* One case, both profiles, three-way equality per element. */
static int checkCase(const char *name, const int16_t *input, const int16_t *coefficients,
                     int expectClamp) {
  static const struct Profile profiles[2] = {
      {"i40", &policy_i40, _mlir_ciface_decimate_i40_batched, _mlir_ciface_decimate_i40_ordered},
      {"i34", &policy_i34, _mlir_ciface_decimate_i34_batched, _mlir_ciface_decimate_i34_ordered},
  };

  int failed = 0;
  for (int index = 0; index < 2; ++index) {
    const struct Profile *profile = &profiles[index];
    int16_t expected[kOutputLength];
    int16_t batched[kOutputLength];
    int16_t ordered[kOutputLength];
    int16_t inputCopy[kInputLength];
    int16_t coefficientCopy[kTapCount];
    int64_t clamped = 0;

    reference(input, coefficients, expected, profile->policy, &clamped);

    for (int64_t position = 0; position < kInputLength; ++position)
      inputCopy[position] = input[position];
    for (int64_t position = 0; position < kTapCount; ++position)
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
    setDescriptor(&inputRef, inputCopy, kInputLength);
    setDescriptor(&coefficientRef, coefficientCopy, kTapCount);
    setDescriptor(&batchedRef, batched, kOutputLength);
    setDescriptor(&orderedRef, ordered, kOutputLength);
    profile->batched(&batchedResult, &inputRef, &coefficientRef, &batchedRef);
    profile->ordered(&orderedResult, &inputRef, &coefficientRef, &orderedRef);

    if (batchedResult.sizes[0] != kOutputLength || orderedResult.sizes[0] != kOutputLength) {
      fprintf(stderr, "%s/%s: result length %lld and %lld, expected %d\n", name, profile->name,
              (long long)batchedResult.sizes[0], (long long)orderedResult.sizes[0], kOutputLength);
      failed = 1;
    }
    for (int64_t position = 0; position < kOutputLength; ++position) {
      const int16_t batchedValue =
          batchedResult.aligned[batchedResult.offset + position * batchedResult.strides[0]];
      const int16_t orderedValue =
          orderedResult.aligned[orderedResult.offset + position * orderedResult.strides[0]];
      if (batchedValue != expected[position] || orderedValue != expected[position]) {
        fprintf(stderr, "%s/%s[%lld]: reference %d, batched %d, ordered %d\n", name, profile->name,
                (long long)position, expected[position], batchedValue, orderedValue);
        failed = 1;
      }
    }
    if (expectClamp && profile->policy == &policy_i34 && clamped == 0) {
      fprintf(stderr,
              "%s/%s: the accumulator never reached its rail, so the case proves "
              "nothing about the clamp\n",
              name, profile->name);
      failed = 1;
    }
    /* The i40 rail is far outside the reachable sum of eight Q15 products, so
     * a clamp there would mean the profile is not what it claims. */
    if (profile->policy == &policy_i40 && clamped != 0) {
      fprintf(stderr, "%s/%s: the i40 accumulator clamped, which this profile cannot reach\n", name,
              profile->name);
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

int main(void) {
  int16_t input[kInputLength];
  int16_t coefficients[kTapCount];
  int failed = 0;

  /* (a) Every sample and every coefficient at the negative rail. Each product
   * is -32768 * -32768 = 2^30 and eight of them are exactly 2^33, one past the
   * i34 maximum, so every lane clamps on its last tap. */
  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = INT16_MIN;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = INT16_MIN;
  failed |= checkCase("all-rail", input, coefficients, /*expectClamp=*/1);

  /* (b) The same rails, but two zero samples every ten positions. A window of
   * eight rails still clamps while any window covering a zero does not, so the
   * clamp now differs BETWEEN lanes of one batched block. */
  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = (position % 10 >= 8) ? 0 : INT16_MIN;
  failed |= checkCase("per-lane rail", input, coefficients, /*expectClamp=*/1);

  /* (c) Alternating sample rails against alternating coefficient rails. Half
   * the products are +2^30 and half are -32768 * 32767, so the running value
   * climbs and falls within one window. */
  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = (position % 2 == 0) ? INT16_MIN : INT16_MAX;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = (tap % 2 == 0) ? INT16_MIN : INT16_MAX;
  failed |= checkCase("alternating rails", input, coefficients, /*expectClamp=*/0);

  /* (d) A single impulse inside the batched range with a ramp of coefficients.
   * Exactly one tap of exactly kTapCount windows is nonzero, which catches a
   * shuffle that picks the wrong lane or the wrong phase. */
  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = 0;
  input[17] = INT16_MIN;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = (int16_t)(1 << (tap + 6));
  failed |= checkCase("single impulse", input, coefficients, /*expectClamp=*/0);

  /* (e) A second impulse in the ordered tail range, so the remainder loop is
   * not exercised only by zeros. */
  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = 0;
  input[39] = INT16_MAX;
  failed |= checkCase("tail impulse", input, coefficients, /*expectClamp=*/0);

  /* (f) All zeros. The exported value is zero everywhere, which pins that the
   * batched loop wrote every element it claimed. */
  for (int64_t position = 0; position < kInputLength; ++position)
    input[position] = 0;
  for (int64_t tap = 0; tap < kTapCount; ++tap)
    coefficients[tap] = INT16_MIN;
  failed |= checkCase("zeros", input, coefficients, /*expectClamp=*/0);

  /* (g) Deterministic xorshift32 trials. The first half draws from the full
   * i16 range so clamps and export saturation both occur; the second half is
   * bounded so the exported values stay inside the destination and remain
   * rounding sensitive. */
  uint32_t state = UINT32_C(0x9e3779b9);
  for (int trial = 0; trial < 16; ++trial) {
    const int wide = trial < 8;
    for (int64_t position = 0; position < kInputLength; ++position) {
      const uint32_t draw = nextRandom(&state);
      input[position] =
          wide ? (int16_t)(draw >> 16) : (int16_t)((int32_t)((draw >> 16) % 4001) - 2000);
    }
    for (int64_t tap = 0; tap < kTapCount; ++tap) {
      const uint32_t draw = nextRandom(&state);
      coefficients[tap] =
          wide ? (int16_t)(draw >> 16) : (int16_t)((int32_t)((draw >> 16) % 4001) - 2000);
    }
    failed |= checkCase(wide ? "random wide" : "random bounded", input, coefficients,
                        /*expectClamp=*/0);
  }

  return failed;
}
