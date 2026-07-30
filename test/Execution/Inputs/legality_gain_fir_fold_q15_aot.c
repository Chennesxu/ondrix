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

extern void _mlir_ciface_negate_then_fir_sharp_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fir_negated_tap_sharp_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_negate_then_fir_hidden_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fir_negated_tap_hidden_q15(MemRefI16 *, MemRefI16 *);

enum { kBatch = 4096, kBatchCount = 16 }; /* 16 * 4096 = the whole i16 domain */

/* The taps the two compiled pairs carry, and the negated taps the illegal
 * real-arithmetic fold produces from them. */
enum { kSharpTap = 16385, kHiddenTap = 9025 };

/* Independent contract arithmetic. None of this reads the compiled code; it
 * decides every output of both programs from the declared semantics. */

/* One ondrix.gain element under nearest_even: exact product, requantization
 * by 15 in explicit floor-division form, i16 saturation. The saturation is
 * the whole story here — it is reachable at exactly one input. */
static int16_t applyGain(int16_t value, int64_t gain) {
  int64_t product = (int64_t)value * gain;
  int64_t quotient = product / 32768;
  int64_t remainder = product % 32768;
  if (remainder < 0) {
    --quotient;
    remainder += 32768;
  }
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
    ++quotient;
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return (int16_t)quotient;
}

/* The filter's export boundary: round half to even from frac 30 to frac 15,
 * then saturate to the Q1.15 destination. */
static int16_t exportQ15(int64_t accumulator) {
  int64_t quotient = accumulator >> 15;
  int64_t remainder = accumulator - (quotient << 15);
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
    ++quotient;
  if (quotient > 32767)
    return 32767;
  if (quotient < -32768)
    return -32768;
  return (int16_t)quotient;
}

/* The single accumulator term of the one-tap filter, on both sides of the
 * candidate rewrite. The certificate compares exactly these two integers. */
static int64_t gainPathTerm(int16_t value, int64_t tap) {
  return (int64_t)applyGain(value, -32768) * tap;
}

static int64_t foldedPathTerm(int16_t value, int64_t tap) { return (int64_t)value * -tap; }

static void runBatch(void (*kernel)(MemRefI16 *, MemRefI16 *), const int16_t *input,
                     int16_t *output, const char *label, int *failed) {
  MemRefI16 inputRef = {(int16_t *)input, (int16_t *)input, 0, {kBatch}, {1}};
  MemRefI16 result;
  kernel(&result, &inputRef);
  if (result.sizes[0] != kBatch) {
    fprintf(stderr, "%s: output length %lld\n", label, (long long)result.sizes[0]);
    *failed = 1;
  }
  for (int64_t i = 0; i < kBatch && i < result.sizes[0]; ++i)
    output[i] = result.aligned[result.offset + i * result.strides[0]];
  free(result.allocated);
}

static void checkAgainstReference(const char *label, int16_t value, int16_t compiled,
                                  int16_t expected, int *failed) {
  if (compiled != expected) {
    fprintf(stderr, "%s at %d: compiled %d, reference %d\n", label, value, compiled, expected);
    *failed = 1;
  }
}

int main(void) {
  int failed = 0;
  /* Divergences between the two programs, counted over the whole domain:
   * at the exact accumulator term the certificate judges, and at the
   * exported sample this particular policy makes observable. */
  int64_t sharpTermDivergences = 0;
  int64_t sharpOutputDivergences = 0;
  int64_t hiddenTermDivergences = 0;
  int64_t hiddenOutputDivergences = 0;
  int sawWitness = 0;

  for (int batch = 0; batch < kBatchCount; ++batch) {
    int16_t input[kBatch];
    int16_t sharpGain[kBatch];
    int16_t sharpFold[kBatch];
    int16_t hiddenGain[kBatch];
    int16_t hiddenFold[kBatch];
    char label[64];
    for (int64_t i = 0; i < kBatch; ++i)
      input[i] = (int16_t)(-32768 + batch * kBatch + i);

    snprintf(label, sizeof label, "sharp gain path batch %d", batch);
    runBatch(_mlir_ciface_negate_then_fir_sharp_q15, input, sharpGain, label, &failed);
    snprintf(label, sizeof label, "sharp folded path batch %d", batch);
    runBatch(_mlir_ciface_fir_negated_tap_sharp_q15, input, sharpFold, label, &failed);
    snprintf(label, sizeof label, "hidden gain path batch %d", batch);
    runBatch(_mlir_ciface_negate_then_fir_hidden_q15, input, hiddenGain, label, &failed);
    snprintf(label, sizeof label, "hidden folded path batch %d", batch);
    runBatch(_mlir_ciface_fir_negated_tap_hidden_q15, input, hiddenFold, label, &failed);

    for (int64_t i = 0; i < kBatch; ++i) {
      int16_t x = input[i];

      int64_t sharpGainTerm = gainPathTerm(x, kSharpTap);
      int64_t sharpFoldTerm = foldedPathTerm(x, kSharpTap);
      int64_t hiddenGainTerm = gainPathTerm(x, kHiddenTap);
      int64_t hiddenFoldTerm = foldedPathTerm(x, kHiddenTap);

      /* Every output of every compiled program is decided independently. */
      checkAgainstReference("sharp gain path", x, sharpGain[i], exportQ15(sharpGainTerm), &failed);
      checkAgainstReference("sharp folded path", x, sharpFold[i], exportQ15(sharpFoldTerm),
                            &failed);
      checkAgainstReference("hidden gain path", x, hiddenGain[i], exportQ15(hiddenGainTerm),
                            &failed);
      checkAgainstReference("hidden folded path", x, hiddenFold[i], exportQ15(hiddenFoldTerm),
                            &failed);

      sharpTermDivergences += sharpGainTerm != sharpFoldTerm;
      sharpOutputDivergences += sharpGain[i] != sharpFold[i];
      hiddenTermDivergences += hiddenGainTerm != hiddenFoldTerm;
      hiddenOutputDivergences += hiddenGain[i] != hiddenFold[i];

      if (x == -32768) {
        /* The only input on which the gain saturates: it returns 32767
         * where the fold assumes 32768. Both taps inherit that one-count
         * difference in the exact term; only the sharp tap survives the
         * export boundary. */
        sawWitness = 1;
        if (sharpGainTerm != 536887295 || sharpFoldTerm != 536903680) {
          fprintf(stderr, "sharp witness terms: gain %lld (want 536887295), fold %lld (want %d)\n",
                  (long long)sharpGainTerm, (long long)sharpFoldTerm, 536903680);
          failed = 1;
        }
        if (sharpGain[i] != 16384 || sharpFold[i] != 16385) {
          fprintf(stderr, "sharp witness outputs: gain %d (want 16384), fold %d (want 16385)\n",
                  sharpGain[i], sharpFold[i]);
          failed = 1;
        }
        if (hiddenGainTerm != 295722175 || hiddenFoldTerm != 295731200) {
          fprintf(stderr, "hidden witness terms: gain %lld (want 295722175), fold %lld (want %d)\n",
                  (long long)hiddenGainTerm, (long long)hiddenFoldTerm, 295731200);
          failed = 1;
        }
        if (hiddenGain[i] != 9025 || hiddenFold[i] != 9025) {
          fprintf(stderr, "hidden witness outputs: gain %d, fold %d (both want 9025)\n",
                  hiddenGain[i], hiddenFold[i]);
          failed = 1;
        }
      }
    }
  }

  if (!sawWitness) {
    fprintf(stderr, "the sweep never reached the saturating input -32768\n");
    failed = 1;
  }
  /* The census the compile-time certificate predicts, replayed on objects.
   * The sharp pair proves this harness can see a one-LSB divergence; the
   * hidden pair proves it is not reporting divergence unconditionally. Move
   * any of these four counts by one and the gate fails. */
  if (sharpTermDivergences != 1 || sharpOutputDivergences != 1) {
    fprintf(stderr, "sharp tap: %lld term divergences, %lld output divergences (want 1 and 1)\n",
            (long long)sharpTermDivergences, (long long)sharpOutputDivergences);
    failed = 1;
  }
  if (hiddenTermDivergences != 1 || hiddenOutputDivergences != 0) {
    fprintf(stderr, "hidden tap: %lld term divergences, %lld output divergences (want 1 and 0)\n",
            (long long)hiddenTermDivergences, (long long)hiddenOutputDivergences);
    failed = 1;
  }
  if (failed)
    return 1;
  printf("gain-into-FIR fold legality gate: %d inputs, sharp %lld/%lld, hidden %lld/%lld\n",
         kBatch * kBatchCount, (long long)sharpTermDivergences, (long long)sharpOutputDivergences,
         (long long)hiddenTermDivergences, (long long)hiddenOutputDivergences);
  return 0;
}
