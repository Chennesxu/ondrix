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

extern void _mlir_ciface_gain_cascade_certified(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_gain_cascade_witness(MemRefI16 *, MemRefI16 *);

enum { kBatch = 4096, kBatchCount = 16 }; /* 16 * 4096 = the whole i16 domain */

/* Independent contract arithmetic: exact product, round-half-even by 15
 * in explicit floor-division form, i16 saturation. */
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

int main(void) {
  int failed = 0;
  int64_t witnessMergeDivergences = 0;
  int64_t swappedMergeDivergences = 0;
  int64_t orderDivergences = 0;

  for (int batch = 0; batch < kBatchCount; ++batch) {
    int16_t input[kBatch];
    int16_t certified[kBatch];
    int16_t witness[kBatch];
    char label[40];
    for (int64_t i = 0; i < kBatch; ++i)
      input[i] = (int16_t)(-32768 + batch * kBatch + i);

    snprintf(label, sizeof label, "certified batch %d", batch);
    runBatch(_mlir_ciface_gain_cascade_certified, input, certified, label, &failed);
    snprintf(label, sizeof label, "witness batch %d", batch);
    runBatch(_mlir_ciface_gain_cascade_witness, input, witness, label, &failed);

    for (int64_t i = 0; i < kBatch; ++i) {
      int16_t x = input[i];

      /* Certified cascade: chain reference AND single merged reference
       * must both match the compiled output — the exhaustive certificate
       * replayed at object level. */
      int16_t chainCertified = applyGain(applyGain(x, 16384), -16384);
      int16_t mergedCertified = applyGain(x, -8192);
      if (certified[i] != chainCertified || chainCertified != mergedCertified) {
        fprintf(stderr, "certified %d: compiled %d, chain %d, merged %d\n", x, certified[i],
                chainCertified, mergedCertified);
        failed = 1;
      }

      /* Witness cascade: the compiled output must equal the two-stage
       * reference; the divergence counters pin why no merge is legal. */
      int16_t chainWitness = applyGain(applyGain(x, 22938), 19661);
      if (witness[i] != chainWitness) {
        fprintf(stderr, "witness %d: compiled %d, chain %d\n", x, witness[i], chainWitness);
        failed = 1;
      }
      int16_t mergedWitness = applyGain(x, 13763); /* q15(22938 * 19661 / 2^15) */
      int16_t swappedWitness = applyGain(applyGain(x, 19661), 22938);
      witnessMergeDivergences += chainWitness != mergedWitness;
      swappedMergeDivergences += swappedWitness != mergedWitness;
      orderDivergences += chainWitness != swappedWitness;
    }
  }

  if (witnessMergeDivergences != 10038 || swappedMergeDivergences != 11418 ||
      orderDivergences != 13556) {
    fprintf(stderr, "divergence counts %lld/%lld/%lld, pinned 10038/11418/13556\n",
            (long long)witnessMergeDivergences, (long long)swappedMergeDivergences,
            (long long)orderDivergences);
    failed = 1;
  }
  return failed;
}
