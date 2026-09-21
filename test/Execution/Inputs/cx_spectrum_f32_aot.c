#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Object gate for the interleaved f32 squared magnitude and magnitude. The
 * profile has no requantization, so every comparison is bit for bit against a
 * reference that walks the same event graph. */

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefF32Rank1;

extern void _mlir_ciface_cx_power_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_power_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_magnitude_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_magnitude_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_rfft64_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_power64_off(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_f32_rfft64_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void _mlir_ciface_cx_magnitude64_fma(MemRefF32Rank1 *, MemRefF32Rank1 *);
extern void ondrix_f32_power_spectrum(const float *, float *);
extern void ondrix_f32_magnitude_spectrum(const float *, float *);

enum { kBins = 8, kElements = 2 * kBins, kPoints = 64, kSpectrumBins = kPoints / 2 + 1 };
enum { kTrialCount = 64 };

static uint32_t floatBits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* The sum seeds on the real square and takes the imaginary one as a single
 * update, which is one add under off and one fused event under fma. */
static float referencePower(float real, float imaginary, int fused) {
  const float seed = real * real;
  return fused ? fmaf(imaginary, imaginary, seed) : seed + imaginary * imaginary;
}

static float referenceMagnitude(float real, float imaginary, int fused) {
  return sqrtf(referencePower(real, imaginary, fused));
}

/* Bin 0 is the named witness: its two contracts disagree in the sum AND
 * survive the root disagreeing, which is what makes this corpus able to tell
 * a fused readout from an unfused one at all. The assertion below proves it
 * rather than assuming it. */
static const float kWitnessReal = -0x1.44961ep-10f;
static const float kWitnessImaginary = -0x1.01e3d2p-10f;

static void fillCorpus(float *values, uint32_t seed) {
  const float directed[kBins][2] = {
      {kWitnessReal, kWitnessImaginary},
      {0.0f, 0.0f},
      {-0.0f, 0.0f},
      {1.0f, 0.0f},
      {0.0f, -1.0f},
      {3.0f, 4.0f},
      {1e-20f, -1e-20f},
      {-2.5f, 0.75f},
  };
  for (int64_t bin = 0; bin < kBins; ++bin) {
    values[2 * bin] = directed[bin][0];
    values[2 * bin + 1] = directed[bin][1];
  }
  if (seed == 0)
    return;
  /* Later trials keep bin 0 and redraw the rest, so the witness runs in every
   * trial and the sweep still moves. */
  for (int64_t bin = 1; bin < kBins; ++bin) {
    for (int64_t half = 0; half < 2; ++half) {
      seed = seed * 1664525u + 1013904223u;
      values[2 * bin + half] = (float)((int32_t)(seed >> 8) - (1 << 23)) * 0x1p-24f;
    }
  }
}

static int checkReadout(const float *values, int fused, int root, const char *label) {
  float input[kElements];
  float output[kBins];
  memcpy(input, values, sizeof(input));
  MemRefF32Rank1 inputRef = {input, input, 0, {kElements}, {1}};
  MemRefF32Rank1 outputRef = {output, output, 0, {kBins}, {1}};
  if (root)
    (fused ? _mlir_ciface_cx_magnitude_fma : _mlir_ciface_cx_magnitude_off)(&inputRef, &outputRef);
  else
    (fused ? _mlir_ciface_cx_power_fma : _mlir_ciface_cx_power_off)(&inputRef, &outputRef);

  int failed = 0;
  for (int64_t bin = 0; bin < kBins; ++bin) {
    const float real = values[2 * bin];
    const float imaginary = values[2 * bin + 1];
    const float expected =
        root ? referenceMagnitude(real, imaginary, fused) : referencePower(real, imaginary, fused);
    if (floatBits(output[bin]) != floatBits(expected)) {
      printf("%s bin %lld: got %a (0x%08x), want %a (0x%08x)\n", label, (long long)bin,
             (double)output[bin], floatBits(output[bin]), (double)expected, floatBits(expected));
      failed = 1;
    }
  }
  return failed;
}

static int checkWitnessDiscriminates(void) {
  float values[kElements];
  fillCorpus(values, 0);
  float powerOff[kBins], powerFma[kBins], magnitudeOff[kBins], magnitudeFma[kBins];
  float input[kElements];
  MemRefF32Rank1 inputRef = {input, input, 0, {kElements}, {1}};
  MemRefF32Rank1 powerOffRef = {powerOff, powerOff, 0, {kBins}, {1}};
  MemRefF32Rank1 powerFmaRef = {powerFma, powerFma, 0, {kBins}, {1}};
  MemRefF32Rank1 magnitudeOffRef = {magnitudeOff, magnitudeOff, 0, {kBins}, {1}};
  MemRefF32Rank1 magnitudeFmaRef = {magnitudeFma, magnitudeFma, 0, {kBins}, {1}};
  memcpy(input, values, sizeof(input));
  _mlir_ciface_cx_power_off(&inputRef, &powerOffRef);
  memcpy(input, values, sizeof(input));
  _mlir_ciface_cx_power_fma(&inputRef, &powerFmaRef);
  memcpy(input, values, sizeof(input));
  _mlir_ciface_cx_magnitude_off(&inputRef, &magnitudeOffRef);
  memcpy(input, values, sizeof(input));
  _mlir_ciface_cx_magnitude_fma(&inputRef, &magnitudeFmaRef);

  int failed = 0;
  if (floatBits(powerOff[0]) == floatBits(powerFma[0])) {
    printf("witness does not separate the power contracts: both %a\n", (double)powerOff[0]);
    failed = 1;
  }
  if (floatBits(magnitudeOff[0]) == floatBits(magnitudeFma[0])) {
    printf("witness does not separate the magnitude contracts: both %a\n", (double)magnitudeOff[0]);
    failed = 1;
  }
  return failed;
}

static void fillSignal(float *signal, uint32_t seed) {
  for (int64_t index = 0; index < kPoints; ++index) {
    seed = seed * 1664525u + 1013904223u;
    signal[index] = (float)((int32_t)(seed >> 8) - (1 << 23)) * 0x1p-24f;
  }
}

/* The source spelling is checked against the staged transform rather than
 * against a second FFT reference: the same bins, read by the same readout. */
static int checkComposed(void) {
  float signal[kPoints], staged[kPoints], spectrum[2 * kSpectrumBins];
  float readout[kSpectrumBins], composed[kSpectrumBins];
  int failed = 0;
  for (uint32_t trial = 1; trial <= kTrialCount; ++trial) {
    for (int root = 0; root < 2; ++root) {
      fillSignal(signal, trial * 2654435761u + (uint32_t)root);
      memcpy(staged, signal, sizeof(staged));
      MemRefF32Rank1 signalRef = {staged, staged, 0, {kPoints}, {1}};
      MemRefF32Rank1 spectrumRef = {spectrum, spectrum, 0, {2 * kSpectrumBins}, {1}};
      MemRefF32Rank1 readoutRef = {readout, readout, 0, {kSpectrumBins}, {1}};
      if (root) {
        _mlir_ciface_f32_rfft64_fma(&signalRef, &spectrumRef);
        _mlir_ciface_cx_magnitude64_fma(&spectrumRef, &readoutRef);
        ondrix_f32_magnitude_spectrum(signal, composed);
      } else {
        _mlir_ciface_f32_rfft64_off(&signalRef, &spectrumRef);
        _mlir_ciface_cx_power64_off(&spectrumRef, &readoutRef);
        ondrix_f32_power_spectrum(signal, composed);
      }
      for (int64_t bin = 0; bin < kSpectrumBins; ++bin) {
        if (floatBits(readout[bin]) != floatBits(composed[bin])) {
          printf("composed %s trial %u bin %lld: staged %a, source %a\n",
                 root ? "magnitude" : "power", trial, (long long)bin, (double)readout[bin],
                 (double)composed[bin]);
          failed = 1;
        }
      }
    }
  }
  return failed;
}

int main(void) {
  int failed = checkWitnessDiscriminates();
  for (uint32_t trial = 0; trial < kTrialCount; ++trial) {
    float values[kElements];
    fillCorpus(values, trial);
    failed |= checkReadout(values, /*fused=*/0, /*root=*/0, "power off");
    failed |= checkReadout(values, /*fused=*/1, /*root=*/0, "power fma");
    failed |= checkReadout(values, /*fused=*/0, /*root=*/1, "magnitude off");
    failed |= checkReadout(values, /*fused=*/1, /*root=*/1, "magnitude fma");
  }
  failed |= checkComposed();
  if (failed) {
    printf("interleaved f32 spectrum readout FAILED\n");
    return 1;
  }
  printf("interleaved f32 spectrum readout OK\n");
  return 0;
}
