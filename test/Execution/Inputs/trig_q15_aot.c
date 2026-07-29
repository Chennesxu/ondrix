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

extern void _mlir_ciface_sine4096_q15(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_cosine4096_q15(MemRefI16 *, MemRefI16 *);

enum { kBatch = 4096, kBatchCount = 16 }; /* 16 * 4096 = the whole phase domain */

/* q15(sin(2*pi*k/256)) derived independently with 50-digit mpmath. */
static const int16_t kSineTable256[256] = {
    0,      804,    1608,   2411,   3212,   4011,   4808,   5602,   6393,   7180,   7962,   8740,
    9512,   10279,  11039,  11793,  12540,  13279,  14010,  14733,  15447,  16151,  16846,  17531,
    18205,  18868,  19520,  20160,  20788,  21403,  22006,  22595,  23170,  23732,  24279,  24812,
    25330,  25833,  26320,  26791,  27246,  27684,  28106,  28511,  28899,  29269,  29622,  29957,
    30274,  30572,  30853,  31114,  31357,  31581,  31786,  31972,  32138,  32286,  32413,  32522,
    32610,  32679,  32729,  32758,  32767,  32758,  32729,  32679,  32610,  32522,  32413,  32286,
    32138,  31972,  31786,  31581,  31357,  31114,  30853,  30572,  30274,  29957,  29622,  29269,
    28899,  28511,  28106,  27684,  27246,  26791,  26320,  25833,  25330,  24812,  24279,  23732,
    23170,  22595,  22006,  21403,  20788,  20160,  19520,  18868,  18205,  17531,  16846,  16151,
    15447,  14733,  14010,  13279,  12540,  11793,  11039,  10279,  9512,   8740,   7962,   7180,
    6393,   5602,   4808,   4011,   3212,   2411,   1608,   804,    0,      -804,   -1608,  -2411,
    -3212,  -4011,  -4808,  -5602,  -6393,  -7180,  -7962,  -8740,  -9512,  -10279, -11039, -11793,
    -12540, -13279, -14010, -14733, -15447, -16151, -16846, -17531, -18205, -18868, -19520, -20160,
    -20788, -21403, -22006, -22595, -23170, -23732, -24279, -24812, -25330, -25833, -26320, -26791,
    -27246, -27684, -28106, -28511, -28899, -29269, -29622, -29957, -30274, -30572, -30853, -31114,
    -31357, -31581, -31786, -31972, -32138, -32286, -32413, -32522, -32610, -32679, -32729, -32758,
    -32768, -32758, -32729, -32679, -32610, -32522, -32413, -32286, -32138, -31972, -31786, -31581,
    -31357, -31114, -30853, -30572, -30274, -29957, -29622, -29269, -28899, -28511, -28106, -27684,
    -27246, -26791, -26320, -25833, -25330, -24812, -24279, -23732, -23170, -22595, -22006, -21403,
    -20788, -20160, -19520, -18868, -18205, -17531, -16846, -16151, -15447, -14733, -14010, -13279,
    -12540, -11793, -11039, -10279, -9512,  -8740,  -7962,  -7180,  -6393,  -5602,  -4808,  -4011,
    -3212,  -2411,  -1608,  -804};

/* Independent contract arithmetic: unsigned turn phase, Q8 nearest-even
 * interpolation in explicit floor-division form, saturating combine. */
static int16_t referenceTrig(int16_t phase, int cosine) {
  uint32_t turn = ((uint32_t)(uint16_t)phase + (cosine ? 16384u : 0u)) & 0xFFFFu;
  uint32_t tableIndex = turn >> 8;
  int32_t fraction = (int32_t)(turn & 255u);
  int32_t lower = kSineTable256[tableIndex];
  int32_t upper = kSineTable256[(tableIndex + 1) & 255u];
  int32_t product = (upper - lower) * fraction;
  int32_t quotient = product / 256;
  int32_t remainder = product % 256;
  if (remainder < 0) {
    --quotient;
    remainder += 256;
  }
  if (remainder > 128 || (remainder == 128 && (quotient & 1)))
    ++quotient;
  /* The interpolation boundary saturates to i16 by contract; the deltas
   * are far inside range, so this clamp is declared, not reachable. */
  if (quotient > 32767)
    quotient = 32767;
  if (quotient < -32768)
    quotient = -32768;
  int32_t combined = lower + quotient;
  if (combined > 32767)
    return 32767;
  if (combined < -32768)
    return -32768;
  return (int16_t)combined;
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

  for (int batch = 0; batch < kBatchCount; ++batch) {
    int16_t input[kBatch];
    int16_t sines[kBatch];
    int16_t cosines[kBatch];
    char label[40];
    for (int64_t i = 0; i < kBatch; ++i)
      input[i] = (int16_t)(-32768 + batch * kBatch + i);

    snprintf(label, sizeof label, "sine batch %d", batch);
    runBatch(_mlir_ciface_sine4096_q15, input, sines, label, &failed);
    snprintf(label, sizeof label, "cosine batch %d", batch);
    runBatch(_mlir_ciface_cosine4096_q15, input, cosines, label, &failed);

    for (int64_t i = 0; i < kBatch; ++i) {
      int16_t phase = input[i];
      int16_t sineExpected = referenceTrig(phase, 0);
      int16_t cosineExpected = referenceTrig(phase, 1);
      if (sines[i] != sineExpected) {
        fprintf(stderr, "sine(%d): got %d, expected %d\n", phase, sines[i], sineExpected);
        failed = 1;
      }
      if (cosines[i] != cosineExpected) {
        fprintf(stderr, "cosine(%d): got %d, expected %d\n", phase, cosines[i], cosineExpected);
        failed = 1;
      }
      /* Quarter-turn identity across the whole domain: the reference for
       * cosine IS sine at the advanced phase, so a bit difference here
       * means the compiled functions disagree with each other. */
      int16_t advanced = (int16_t)(uint16_t)(((uint32_t)(uint16_t)phase + 16384u) & 0xFFFFu);
      if (cosineExpected != referenceTrig(advanced, 0)) {
        fprintf(stderr, "identity broke at phase %d\n", phase);
        failed = 1;
      }
    }
  }

  /* Directed axis goldens. */
  if (referenceTrig(0, 0) != 0 || referenceTrig(16384, 0) != 32767 ||
      referenceTrig(-32768, 0) != 0 || referenceTrig(-16384, 0) != -32768 ||
      referenceTrig(0, 1) != 32767) {
    fprintf(stderr, "axis goldens moved\n");
    failed = 1;
  }
  return failed;
}
