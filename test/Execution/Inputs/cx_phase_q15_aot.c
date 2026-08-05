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

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

extern void _mlir_ciface_cx_phase_q15(MemRefI16 *, MemRefI32 *);

enum { kBlock = 4096 };

static int failures;
static int32_t arctangentTable[129];

/* Independent reference: the declared table regenerated from its definition
 * and the declared arithmetic written out, so a table that stops being the
 * contract's table disagrees here. */
static void buildTable(void) {
  const double twoPi = 6.28318530717958647692528676655900577;
  for (int k = 0; k <= 128; ++k)
    arctangentTable[k] = k == 128 ? 8192 : (int32_t)nearbyint(atan(k / 128.0) / twoPi * 65536.0);
}

static int32_t roundHalfEven(int32_t value, int shift) {
  int32_t quotient = value >> shift;
  int32_t remainder = value - (quotient << shift);
  int32_t half = (int32_t)1 << (shift - 1);
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return quotient;
}

static uint16_t referencePhase(int32_t real, int32_t imaginary) {
  if (real == 0 && imaginary == 0)
    return 0;
  int32_t a = real < 0 ? -real : real;
  int32_t b = imaginary < 0 ? -imaginary : imaginary;
  int swapped = b > a;
  int32_t high = a > b ? a : b;
  int32_t low = a > b ? b : a;
  int64_t numerator = (int64_t)low << 16;
  int64_t quotient = numerator / high;
  int64_t remainder = numerator - quotient * high;
  if (2 * remainder > high || (2 * remainder == high && (quotient & 1)))
    ++quotient;
  int32_t ratio = (int32_t)quotient;
  int32_t index = ratio >> 9;
  if (index > 127)
    index = 127;
  int32_t fraction = ratio - (index << 9);
  int32_t base = arctangentTable[index] +
                 roundHalfEven((arctangentTable[index + 1] - arctangentTable[index]) * fraction, 9);
  int32_t folded = swapped ? 16384 - base : base;
  int32_t turn;
  if (real >= 0)
    turn = imaginary >= 0 ? folded : -folded;
  else
    turn = imaginary >= 0 ? 32768 - folded : 32768 + folded;
  return (uint16_t)(turn & 0xFFFF);
}

static int32_t pack(int16_t real, int16_t imaginary) {
  return (int32_t)(((uint32_t)(uint16_t)imaginary << 16) | (uint32_t)(uint16_t)real);
}

static void checkBatch(const int32_t *packed, int64_t count, const char *label) {
  MemRefI32 inputRef = {(int32_t *)packed, (int32_t *)packed, 0, {count}, {1}};
  MemRefI16 output;
  _mlir_ciface_cx_phase_q15(&output, &inputRef);
  for (int64_t i = 0; i < count; ++i) {
    int16_t real = (int16_t)(packed[i] & 0xFFFF);
    int16_t imaginary = (int16_t)((uint32_t)packed[i] >> 16);
    uint16_t expected = referencePhase(real, imaginary);
    uint16_t got = (uint16_t)output.aligned[output.offset + i];
    if (got != expected && failures++ < 8)
      fprintf(stderr, "%s: phase(%d, %d) got %u, expected %u\n", label, real, imaginary, got,
              expected);
  }
  free(output.allocated);
}

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

/* The eight octants meet at the axes and the diagonals, and the fold is
 * exact there by construction rather than by rounding. These are the values
 * the contract names, so they are asserted directly and not merely covered. */
static void checkStructuralAngles(void) {
  struct {
    int16_t real;
    int16_t imaginary;
    uint16_t turn;
    const char *name;
  } cases[] = {
      {0, 0, 0, "origin"},
      {1, 0, 0, "east"},
      {32767, 0, 0, "east rail"},
      {0, 1, 16384, "north"},
      {0, 32767, 16384, "north rail"},
      {-1, 0, 32768, "west"},
      {-32768, 0, 32768, "west rail"},
      {0, -1, 49152, "south"},
      {0, -32768, 49152, "south rail"},
      {1, 1, 8192, "northeast"},
      {-1, 1, 24576, "northwest"},
      {-1, -1, 40960, "southwest"},
      {1, -1, 57344, "southeast"},
      {32767, 32767, 8192, "northeast rail"},
      {-32768, -32768, 40960, "southwest rail"},
  };
  int32_t packed[sizeof cases / sizeof cases[0]];
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i)
    packed[i] = pack(cases[i].real, cases[i].imaginary);
  MemRefI32 inputRef = {packed, packed, 0, {(int64_t)(sizeof cases / sizeof cases[0])}, {1}};
  MemRefI16 output;
  _mlir_ciface_cx_phase_q15(&output, &inputRef);
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
    uint16_t got = (uint16_t)output.aligned[output.offset + (int64_t)i];
    if (got != cases[i].turn) {
      fprintf(stderr, "%s: phase(%d, %d) got %u, contract says %u\n", cases[i].name, cases[i].real,
              cases[i].imaginary, got, cases[i].turn);
      ++failures;
    }
  }
  free(output.allocated);
}

int main(void) {
  buildTable();
  checkStructuralAngles();

  /* The domain is 2^32 pairs, so the sweep is structured rather than
   * exhaustive and says what it covers: every value of one component
   * against several values of the other, all four axes and all four
   * diagonals over their whole range, and a deterministic random sweep. */
  int32_t packed[kBlock];
  static const int16_t pivots[] = {1, -1, 3, -3, 181, -181, 32767, -32768, 16384, -16384};
  for (size_t p = 0; p < sizeof pivots / sizeof pivots[0]; ++p) {
    for (int64_t block = 0; block < 16; ++block) {
      for (int64_t i = 0; i < kBlock; ++i) {
        int16_t moving = (int16_t)(int32_t)(block * kBlock + i - 32768);
        packed[i] = pack(pivots[p], moving);
      }
      checkBatch(packed, kBlock, "imaginary sweep");
      for (int64_t i = 0; i < kBlock; ++i) {
        int16_t moving = (int16_t)(int32_t)(block * kBlock + i - 32768);
        packed[i] = pack(moving, pivots[p]);
      }
      checkBatch(packed, kBlock, "real sweep");
    }
  }

  for (int64_t block = 0; block < 16; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t v = (int16_t)(int32_t)(block * kBlock + i - 32768);
      packed[i] = pack(v, v);
    }
    checkBatch(packed, kBlock, "diagonal");
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t v = (int16_t)(int32_t)(block * kBlock + i - 32768);
      packed[i] = pack(v, (int16_t)-v);
    }
    checkBatch(packed, kBlock, "antidiagonal");
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t v = (int16_t)(int32_t)(block * kBlock + i - 32768);
      packed[i] = pack(v, 0);
    }
    checkBatch(packed, kBlock, "real axis");
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t v = (int16_t)(int32_t)(block * kBlock + i - 32768);
      packed[i] = pack(0, v);
    }
    checkBatch(packed, kBlock, "imaginary axis");
  }

  uint32_t state = 0x6C1D93AFu;
  for (int64_t block = 0; block < 256; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      state = nextState(state);
      packed[i] = (int32_t)state;
    }
    checkBatch(packed, kBlock, "random");
  }

  /* Accuracy against the mathematical argument is a property of the
   * declared table, not part of the contract, so it is measured and
   * reported rather than asserted as an equality. */
  double worst = 0.0;
  const double twoPi = 6.28318530717958647692528676655900577;
  state = 0x1AE45C07u;
  for (int64_t trial = 0; trial < 200000; ++trial) {
    state = nextState(state);
    int16_t real = (int16_t)(state & 0xFFFF);
    state = nextState(state);
    int16_t imaginary = (int16_t)(state & 0xFFFF);
    if (real == 0 && imaginary == 0)
      continue;
    double exact = atan2((double)imaginary, (double)real) / twoPi * 65536.0;
    if (exact < 0.0)
      exact += 65536.0;
    double got = (double)referencePhase(real, imaginary);
    double error = fabs(got - exact);
    if (error > 32768.0)
      error = 65536.0 - error;
    if (error > worst)
      worst = error;
  }
  printf("cx_phase: worst deviation from the exact argument %.3f raw turn units\n", worst);
  if (worst > 4.0) {
    fprintf(stderr, "cx_phase table accuracy regressed past four turn units\n");
    ++failures;
  }
  return failures != 0;
}
