#include <math.h>
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

extern void _mlir_ciface_log2_q0_16(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_exp2_q5_11(MemRefI16 *, MemRefI16 *);

enum { kBlock = 4096, kBlocks = 16 };

static int failures;

/* Independent reference: the declared table and the declared interpolation,
 * regenerated here from the mathematical definition rather than copied from
 * the compiler's constant. If the emitted table ever stops being the table
 * the contract names, this disagrees. */
static int32_t logTable[129];
static int32_t expTable[129];

static void buildTables(void) {
  for (int k = 0; k <= 128; ++k) {
    logTable[k] = k == 128 ? 2048 : (int32_t)nearbyint(log2(1.0 + k / 128.0) * 2048.0);
    expTable[k] = k == 128 ? 65536 : (int32_t)nearbyint(exp2(k / 128.0) * 32768.0);
  }
}

static int32_t roundHalfEven(int32_t value, int shift) {
  if (shift == 0)
    return value;
  int32_t quotient = value >> shift;
  int32_t remainder = value - (quotient << shift);
  int32_t half = (int32_t)1 << (shift - 1);
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return quotient;
}

static int16_t referenceLog2(uint16_t raw) {
  if (raw == 0)
    return -32768;
  int exponent = 15;
  while (!((raw >> exponent) & 1))
    --exponent;
  int32_t mantissa = (int32_t)raw << (15 - exponent);
  int index = (mantissa >> 8) & 127;
  int32_t fraction = mantissa & 255;
  int32_t interpolated = roundHalfEven((logTable[index + 1] - logTable[index]) * fraction, 8);
  return (int16_t)((exponent - 16) * 2048 + logTable[index] + interpolated);
}

static uint16_t referenceExp2(int16_t value) {
  if (value >= 0)
    return 65535;
  int32_t exponent = value >> 11;
  int32_t fraction = value & 2047;
  int index = fraction >> 4;
  int32_t interpolant = fraction & 15;
  int32_t mantissa =
      expTable[index] + roundHalfEven((expTable[index + 1] - expTable[index]) * interpolant, 4);
  return (uint16_t)roundHalfEven(mantissa, (int)(-1 - exponent));
}

int main(void) {
  buildTables();
  int16_t input[kBlock];
  /* Both domains are 2^16 values, so both are swept whole: the pole, the
   * ceiling, every binade boundary and every interpolation tie are reached
   * by construction. */
  for (int64_t block = 0; block < kBlocks; ++block) {
    for (int64_t i = 0; i < kBlock; ++i)
      input[i] = (int16_t)(int32_t)(block * kBlock + i - 32768);
    MemRefI16 inputRef = {input, input, 0, {kBlock}, {1}};

    MemRefI16 logs;
    _mlir_ciface_log2_q0_16(&logs, &inputRef);
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t expected = referenceLog2((uint16_t)input[i]);
      int16_t got = logs.aligned[logs.offset + i];
      if (got != expected && failures++ < 8)
        fprintf(stderr, "log2(%u): got %d, expected %d\n", (unsigned)(uint16_t)input[i], got,
                expected);
    }
    free(logs.allocated);

    MemRefI16 exps;
    _mlir_ciface_exp2_q5_11(&exps, &inputRef);
    for (int64_t i = 0; i < kBlock; ++i) {
      uint16_t expected = referenceExp2(input[i]);
      uint16_t got = (uint16_t)exps.aligned[exps.offset + i];
      if (got != expected && failures++ < 8)
        fprintf(stderr, "exp2(%d): got %u, expected %u\n", input[i], got, expected);
    }
    free(exps.allocated);
  }

  /* Directed values that name what the contract declares rather than what a
   * sweep happens to cover. */
  struct {
    uint16_t input;
    int16_t expected;
  } logGoldens[] = {
      {0, -32768},    /* the declared pole */
      {1, -32768},    /* 2^-16 exactly */
      {32768, -2048}, /* one half: -1 in Q5.11 */
      {65535, 0},     /* just under one: rounds up to zero */
      {16384, -4096}, /* one quarter: -2 */
  };
  for (size_t i = 0; i < sizeof logGoldens / sizeof logGoldens[0]; ++i) {
    int16_t got = referenceLog2(logGoldens[i].input);
    if (got != logGoldens[i].expected) {
      fprintf(stderr, "log2 golden %u: reference gives %d, contract says %d\n",
              (unsigned)logGoldens[i].input, got, logGoldens[i].expected);
      ++failures;
    }
  }
  struct {
    int16_t input;
    uint16_t expected;
  } expGoldens[] = {
      {0, 65535},     /* the declared ceiling */
      {1, 65535},     /* above the range */
      {-2048, 32768}, /* 2^-1 exactly */
      {-4096, 16384}, /* 2^-2 exactly */
      {-32768, 1},    /* 2^-16 exactly */
  };
  for (size_t i = 0; i < sizeof expGoldens / sizeof expGoldens[0]; ++i) {
    uint16_t got = referenceExp2(expGoldens[i].input);
    if (got != expGoldens[i].expected) {
      fprintf(stderr, "exp2 golden %d: reference gives %u, contract says %u\n", expGoldens[i].input,
              got, expGoldens[i].expected);
      ++failures;
    }
  }

  /* The round trip is not the identity and the contract never said it was;
   * what it owes is a bound. The table's linear interpolation error is under
   * 0.03 Q11 units (the segment is 2^-7 wide and log2'' is at most 1/ln2),
   * so each direction's rounding dominates and the logarithm is within about
   * one Q11 unit of the exact one. Two Q11 units of logarithm is a relative
   * magnitude change of 2*ln2/2048, and one more raw unit covers exp2's own
   * rounding. Anything past that is a defect, not the declared cost. */
  int64_t worst = 0;
  for (int32_t raw = 1; raw <= 65535; ++raw) {
    int16_t logged = referenceLog2((uint16_t)raw);
    int64_t back = referenceExp2(logged);
    int64_t quantum = 1 + (int64_t)ceil(raw * 2.0 * M_LN2 / 2048.0);
    int64_t error = back - raw;
    if (error < 0)
      error = -error;
    if (error > quantum) {
      if (failures++ < 8)
        fprintf(stderr, "round trip %d: back %lld, allowed %lld\n", raw, (long long)back,
                (long long)quantum);
    }
    if (error > worst)
      worst = error;
  }
  printf("log2/exp2 round trip: worst discrepancy %lld raw units over 65535 magnitudes\n",
         (long long)worst);
  return failures != 0;
}
