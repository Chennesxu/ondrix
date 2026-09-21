#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for the interleaved f32 phase. Three separable claims: the
 * emitted turn is bit for bit the declared event graph, it stays within two
 * binary32 steps of the exact argument, and it is exact on the axes and
 * diagonals. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

extern void _mlir_ciface_cx_phase_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_phase_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_rfft64_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_phase64_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void ondrix_f32_phase_spectrum(const float *, float *);

enum { kBins = 1024, kElements = 2 * kBins, kTrialCount = 48 };
enum { kPoints = 64, kSpectrumBins = kPoints / 2 + 1 };

/* The bound the contract states, two binary32 steps of the turn's own top
 * decade. The measured maximum is 0.98 of one step; the reachable budget is
 * wider because the unfold alone spends half a step in [0.5, 1). */
static const double kTurnBound = 2.0 * 0x1p-24;

static const float kCoefficients[9] = {0x1.45f306p-3f,  -0x1.b2988p-5f,  0x1.04a9cap-5f,
                                       -0x1.725f9cp-6f, 0x1.1578fp-6f,   -0x1.875cf8p-7f,
                                       0x1.bd49e8p-8f,  -0x1.4f342ap-9f, 0x1.db9b5ap-12f};

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* The declared event graph, in emission order. Only the ratio and the Horner
 * chain round; every fold and unfold constant is a power of two. */
static float referenceTurn(float real, float imaginary, int fused) {
  const float a = fabsf(real), b = fabsf(imaginary);
  const int folded = b > a;
  const float high = folded ? b : a;
  const float low = folded ? a : b;
  const float denominator = high == 0.0f ? 1.0f : high;
  const float ratio = low / denominator;
  const float square = ratio * ratio;
  float accumulator = kCoefficients[8];
  for (int index = 7; index >= 0; --index)
    accumulator = fused ? fmaf(square, accumulator, kCoefficients[index])
                        : kCoefficients[index] + square * accumulator;
  const float eighth = ratio * accumulator;
  const float octant = folded ? 0.25f - eighth : eighth;
  float reflected = 1.0f - octant;
  if (reflected == 1.0f)
    reflected = 0.0f;
  const float right = imaginary >= 0.0f ? octant : reflected;
  const float left = imaginary >= 0.0f ? 0.5f - octant : 0.5f + octant;
  return real >= 0.0f ? right : left;
}

/* The turn is circular, so a result just below one and a reference just above
 * zero are the same angle. */
static double turnDistance(double got, double want) {
  double direct = fabs(got - want);
  return direct < 1.0 - direct ? direct : 1.0 - direct;
}

static double exactTurn(float real, float imaginary) {
  if (real == 0.0f && imaginary == 0.0f)
    return 0.0;
  double turn = atan2((double)imaginary, (double)real) / (2.0 * M_PI);
  turn = turn - floor(turn);
  return turn >= 1.0 ? 0.0 : turn;
}

static uint32_t nextRandom(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static float randomScaled(uint32_t *state) {
  const float mantissa = (float)((int32_t)(nextRandom(state) >> 8) - (1 << 23)) * 0x1p-24f;
  const int exponent = (int)(nextRandom(state) % 61u) - 30;
  return ldexpf(mantissa, exponent);
}

static void fillCorpus(float *values, uint32_t trial) {
  uint32_t state = trial * 2654435761u + 12345u;
  for (int64_t bin = 0; bin < kBins; ++bin) {
    values[2 * bin] = randomScaled(&state);
    values[2 * bin + 1] = randomScaled(&state);
  }
  /* The named structure every trial carries: the origin, the four axes and
   * the four diagonals, whose turns the contract claims are exact. */
  const float unit = ldexpf(1.0f, (int)(trial % 40u) - 20);
  const float structured[9][2] = {{0.0f, 0.0f},  {unit, 0.0f},  {0.0f, unit},
                                  {-unit, 0.0f}, {0.0f, -unit}, {unit, unit},
                                  {unit, -unit}, {-unit, unit}, {-unit, -unit}};
  for (int64_t index = 0; index < 9; ++index) {
    values[2 * index] = structured[index][0];
    values[2 * index + 1] = structured[index][1];
  }
}

static int checkTrial(uint32_t trial, double *worst) {
  float input[kElements], output[kBins];
  float values[kElements];
  fillCorpus(values, trial);
  int failed = 0;
  for (int fused = 0; fused < 2; ++fused) {
    memcpy(input, values, sizeof(input));
    MemRefF32Rank1 inputRef = {input, input, 0, {kElements}, {1}};
    MemRefF32Rank1 outputRef = {output, output, 0, {kBins}, {1}};
    (fused ? _mlir_ciface_cx_phase_fma : _mlir_ciface_cx_phase_off)(&inputRef, &outputRef);
    for (int64_t bin = 0; bin < kBins; ++bin) {
      const float real = values[2 * bin], imaginary = values[2 * bin + 1];
      const float expected = referenceTurn(real, imaginary, fused);
      if (floatBits(output[bin]) != floatBits(expected)) {
        printf("phase %s trial %u bin %lld: got %a, want %a\n", fused ? "fma" : "off", trial,
               (long long)bin, (double)output[bin], (double)expected);
        failed = 1;
      }
      if (output[bin] < 0.0f || output[bin] >= 1.0f) {
        printf("phase %s trial %u bin %lld: %a is outside the half-open turn\n",
               fused ? "fma" : "off", trial, (long long)bin, (double)output[bin]);
        failed = 1;
      }
      const double distance = turnDistance((double)output[bin], exactTurn(real, imaginary));
      if (distance > *worst)
        *worst = distance;
      if (distance > kTurnBound) {
        printf("phase %s trial %u bin %lld: re=%a im=%a is %g turns out\n", fused ? "fma" : "off",
               trial, (long long)bin, (double)real, (double)imaginary, distance);
        failed = 1;
      }
      /* The first nine bins are the structured ones, and the contract claims
       * the axes and diagonals exactly rather than within the bound. */
      if (bin < 9 && distance != 0.0) {
        printf("phase %s trial %u structured bin %lld: re=%a im=%a is not exact (%g)\n",
               fused ? "fma" : "off", trial, (long long)bin, (double)real, (double)imaginary,
               distance);
        failed = 1;
      }
    }
  }
  return failed;
}

/* The source spelling is checked against the staged transform rather than
 * against a second FFT reference. */
static int checkComposed(void) {
  float signal[kPoints], staged[kPoints], spectrum[2 * kSpectrumBins];
  float readout[kSpectrumBins], composed[kSpectrumBins];
  int failed = 0;
  for (uint32_t trial = 1; trial <= 32; ++trial) {
    uint32_t state = trial * 40503u + 7u;
    for (int64_t index = 0; index < kPoints; ++index)
      signal[index] = (float)((int32_t)(nextRandom(&state) >> 8) - (1 << 23)) * 0x1p-24f;
    memcpy(staged, signal, sizeof(staged));
    MemRefF32Rank1 signalRef = {staged, staged, 0, {kPoints}, {1}};
    MemRefF32Rank1 spectrumRef = {spectrum, spectrum, 0, {2 * kSpectrumBins}, {1}};
    MemRefF32Rank1 readoutRef = {readout, readout, 0, {kSpectrumBins}, {1}};
    _mlir_ciface_f32_rfft64_off(&signalRef, &spectrumRef);
    _mlir_ciface_cx_phase64_off(&spectrumRef, &readoutRef);
    ondrix_f32_phase_spectrum(signal, composed);
    for (int64_t bin = 0; bin < kSpectrumBins; ++bin) {
      if (floatBits(readout[bin]) != floatBits(composed[bin])) {
        printf("composed phase trial %u bin %lld: staged %a, source %a\n", trial, (long long)bin,
               (double)readout[bin], (double)composed[bin]);
        failed = 1;
      }
    }
  }
  return failed;
}

int main(void) {
  int failed = 0;
  double worst = 0.0;
  for (uint32_t trial = 0; trial < kTrialCount; ++trial)
    failed |= checkTrial(trial, &worst);
  failed |= checkComposed();
  if (failed) {
    printf("interleaved f32 phase FAILED\n");
    return 1;
  }
  printf("interleaved f32 phase OK, worst %g turns (%.3f x 2^-24) over %d pairs\n", worst,
         worst / 0x1p-24, 2 * kTrialCount * kBins);
  return 0;
}
