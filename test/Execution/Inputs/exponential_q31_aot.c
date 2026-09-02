#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_log2_q0_32(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_exp2_q6_26(MemRefI32 *, MemRefI32 *);

enum { kBlock = 4096, kMaxDistance = 1 };

static int failures;
static int64_t worstLog, worstExp;

/* Independent reference: the two declared tables regenerated from their
 * definitions and the declared programs written out, so a table or a term
 * that stops being the contract's disagrees here. */
static int32_t logTable[1024];
static uint32_t expTable[1024];

static void buildTables(void) {
  for (int k = 0; k < 1024; ++k) {
    logTable[k] = (int32_t)nearbyint(log2(1.0 + k / 1024.0) * 1073741824.0);
    expTable[k] = (uint32_t)nearbyint(exp2(k / 1024.0) * 2147483648.0);
  }
}

static int64_t roundHalfEven(uint64_t value, int shift) {
  uint64_t quotient = value >> shift;
  uint64_t remainder = value - (quotient << shift);
  uint64_t half = (uint64_t)1 << (shift - 1);
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return (int64_t)quotient;
}

static int32_t referenceLog2(uint32_t u) {
  if (u == 0)
    return INT32_MIN;
  int exponent = 31;
  while (!((u >> exponent) & 1))
    --exponent;
  uint64_t mantissa = (uint64_t)u << (31 - exponent);
  uint64_t coarse = mantissa >> 21;
  uint64_t base = coarse << 21;
  uint64_t ratio = ((mantissa - base) << 32) / base;
  uint64_t series = ratio * 24204406u - ((ratio * ratio) >> 22) * 11819u;
  uint64_t total = ((uint64_t)logTable[coarse - 1024] << 26) + series;
  return (int32_t)(((int64_t)(exponent - 32) << 26) + roundHalfEven(total, 30));
}

static uint32_t referenceExp2(int32_t v) {
  if (v >= 0)
    return 0xFFFFFFFFu;
  int32_t exponent = v >> 26;
  uint32_t fraction = (uint32_t)v & ((1u << 26) - 1u);
  uint64_t angle = ((uint64_t)(fraction & 0xFFFFu) * 762123384786ull) >> 26;
  uint64_t square = (angle * angle) >> 40;
  uint64_t cube = (square * angle) >> 40;
  uint64_t series = angle + (square >> 1) + cube / 6;
  uint64_t word = expTable[fraction >> 16];
  uint64_t wide = (word << 31) + ((word * series) >> 9);
  return (uint32_t)roundHalfEven(wide, 30 - exponent);
}

static int64_t nearestLog2(uint32_t u) {
  return (int64_t)roundl(log2l((long double)u / 4294967296.0L) * 67108864.0L);
}

static int64_t nearestExp2(int32_t v) {
  return (int64_t)roundl(exp2l((long double)v / 67108864.0L) * 4294967296.0L);
}

static void report(const char *label, int64_t input, int64_t got, int64_t expected) {
  if (failures++ < 8)
    fprintf(stderr, "%s(%lld): got %lld, expected %lld\n", label, (long long)input, (long long)got,
            (long long)expected);
}

static void checkBlock(const uint32_t *magnitudes, const int32_t *exponents) {
  int32_t input[kBlock];
  for (int64_t i = 0; i < kBlock; ++i)
    input[i] = (int32_t)magnitudes[i];
  MemRefI32 inputRef = {input, input, 0, {kBlock}, {1}};
  MemRefI32 logs;
  _mlir_ciface_log2_q0_32(&logs, &inputRef);
  for (int64_t i = 0; i < kBlock; ++i) {
    int32_t got = logs.aligned[logs.offset + i];
    if (got != referenceLog2(magnitudes[i]))
      report("log2_contract", magnitudes[i], got, referenceLog2(magnitudes[i]));
    if (magnitudes[i] != 0) {
      int64_t distance = llabs((int64_t)got - nearestLog2(magnitudes[i]));
      if (distance > worstLog)
        worstLog = distance;
      if (distance > kMaxDistance)
        report("log2_distance", magnitudes[i], got, nearestLog2(magnitudes[i]));
    }
  }
  free(logs.allocated);

  for (int64_t i = 0; i < kBlock; ++i)
    input[i] = exponents[i];
  MemRefI32 exps;
  _mlir_ciface_exp2_q6_26(&exps, &inputRef);
  for (int64_t i = 0; i < kBlock; ++i) {
    uint32_t got = (uint32_t)exps.aligned[exps.offset + i];
    if (got != referenceExp2(exponents[i]))
      report("exp2_contract", exponents[i], got, referenceExp2(exponents[i]));
    if (exponents[i] < 0) {
      int64_t distance = llabs((int64_t)got - nearestExp2(exponents[i]));
      if (distance > worstExp)
        worstExp = distance;
      if (distance > kMaxDistance)
        report("exp2_distance", exponents[i], got, nearestExp2(exponents[i]));
    }
  }
  free(exps.allocated);
}

int main(void) {
  buildTables();
  uint32_t magnitudes[kBlock];
  int32_t exponents[kBlock];

  /* Every table point at both residual rails and the residual midpoint, in
   * the top binade for log2 and at four exponents for exp2, plus the pole,
   * the ceiling and every binade boundary. */
  for (int64_t block = 0; block < 4; ++block) {
    for (int64_t i = 0; i < kBlock; i += 4) {
      uint32_t coarse = 1024u + (uint32_t)(block * (kBlock / 4) + i / 4);
      magnitudes[i + 0] = coarse << 21;
      magnitudes[i + 1] = (coarse << 21) | 1u;
      magnitudes[i + 2] = (coarse << 21) | (1u << 20);
      magnitudes[i + 3] = (coarse << 21) | ((1u << 21) - 1u);
      static const int32_t binades[4] = {-1, -2, -17, -32};
      int32_t base = (int32_t)((uint32_t)binades[(block + i / 4) & 3] << 26);
      uint32_t k = (uint32_t)(block * (kBlock / 4) + i / 4);
      exponents[i + 0] = base + (int32_t)(k << 16);
      exponents[i + 1] = base + (int32_t)((k << 16) | 1u);
      exponents[i + 2] = base + (int32_t)((k << 16) | (1u << 15));
      exponents[i + 3] = base + (int32_t)((k << 16) | 0xFFFFu);
    }
    checkBlock(magnitudes, exponents);
  }
  for (int64_t i = 0; i < kBlock; ++i) {
    int64_t bit = i % 33;
    magnitudes[i] = bit == 32 ? 0u : (1u << bit) - (uint32_t)(i / 33 % 2);
    exponents[i] = (int32_t)(((int64_t)(i % 66) - 33) << 26) + (int32_t)(i / 66 % 3) - 1;
  }
  checkBlock(magnitudes, exponents);

  /* A stride walk coprime with 2^32, so 2^22 values spread over the whole
   * domain rather than clustering; the exponent arm walks the negative half. */
  uint32_t state = 0u;
  for (int64_t block = 0; block < 1024; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      state += 1021u * 1048573u;
      magnitudes[i] = state;
      exponents[i] = (int32_t)(state | 0x80000000u);
    }
    checkBlock(magnitudes, exponents);
  }

  /* Directed values that name what the contract declares. */
  struct {
    uint32_t input;
    int32_t expected;
  } logGoldens[] = {
      {0u, INT32_MIN},                             /* the declared pole */
      {1u, INT32_MIN},                             /* 2^-32 exactly: -32 */
      {2147483648u, -67108864},                    /* one half: -1 */
      {1073741824u, -134217728}, {4294967295u, 0}, /* just under one rounds up to zero */
  };
  for (size_t i = 0; i < sizeof logGoldens / sizeof logGoldens[0]; ++i)
    if (referenceLog2(logGoldens[i].input) != logGoldens[i].expected)
      report("log2_golden", logGoldens[i].input, referenceLog2(logGoldens[i].input),
             logGoldens[i].expected);
  struct {
    int32_t input;
    uint32_t expected;
  } expGoldens[] = {
      {0, 0xFFFFFFFFu},          /* the declared ceiling */
      {1, 0xFFFFFFFFu},          /* above the range */
      {-67108864, 2147483648u},  /* 2^-1 exactly */
      {-134217728, 1073741824u}, /* 2^-2 exactly */
      {INT32_MIN, 1u},           /* 2^-32 exactly */
  };
  for (size_t i = 0; i < sizeof expGoldens / sizeof expGoldens[0]; ++i)
    if (referenceExp2(expGoldens[i].input) != expGoldens[i].expected)
      report("exp2_golden", expGoldens[i].input, referenceExp2(expGoldens[i].input),
             expGoldens[i].expected);

  /* The round trip is not the identity; what it owes is a bound. Each
   * direction is within one LSB of correctly rounded, so the logarithm is
   * within about 1.5 Q26 units and the magnitude comes back within that
   * relative change plus exp2's own two units. */
  int64_t worstTrip = 0;
  state = 0x9E3779B9u;
  for (int64_t trial = 0; trial < 1 << 20; ++trial) {
    state += 1021u * 1048573u;
    uint32_t raw = state ? state : 1u;
    int64_t back = referenceExp2(referenceLog2(raw));
    int64_t quantum = 2 + (int64_t)ceil((double)raw * 3.0 * M_LN2 / 67108864.0);
    int64_t error = llabs(back - (int64_t)raw);
    if (error > quantum)
      report("round_trip", raw, back, raw);
    if (error > worstTrip)
      worstTrip = error;
  }

  printf("log2/exp2 Q31: worst distance %lld/%lld LSB, worst round trip %lld raw units\n",
         (long long)worstLog, (long long)worstExp, (long long)worstTrip);
  return failures != 0;
}
