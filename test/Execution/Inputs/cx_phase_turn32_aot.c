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
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI64;

extern void _mlir_ciface_phase_q15_turn32(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_phase_q31_turn32(MemRefI32 *, MemRefI64 *);

enum { kBlock = 4096, kMaxDistance = 1 };
static const uint64_t kInverseTwoPiQ40 = 174992710548ull;

static int failures;
static int64_t worstDistance;
static int64_t arctangentTable[1025];

/* Independent reference: the table regenerated from its definition, the
 * ratio, the angle addition and the series written out from the contract. */
static void buildTable(void) {
  const double twoPi = 6.28318530717958647692528676655900577;
  for (int k = 0; k <= 1024; ++k)
    arctangentTable[k] =
        k == 1024 ? (int64_t)1 << 30 : (int64_t)nearbyint(atan(k / 1024.0) / twoPi * 8589934592.0);
}

static uint32_t referenceTurn(int64_t real, int64_t imaginary) {
  if (real == 0 && imaginary == 0)
    return 0u;
  uint64_t a = real < 0 ? (uint64_t)-real : (uint64_t)real;
  uint64_t b = imaginary < 0 ? (uint64_t)-imaginary : (uint64_t)imaginary;
  int swapped = b > a;
  uint64_t high = a > b ? a : b;
  uint64_t low = a > b ? b : a;
  uint64_t numerator = low << 32;
  uint64_t ratio = numerator / high;
  uint64_t remainder = numerator - ratio * high;
  if (2 * remainder > high || (2 * remainder == high && (ratio & 1)))
    ++ratio;
  uint64_t coarse = ratio >> 22;
  uint64_t residual = ratio - (coarse << 22);
  uint64_t u = (residual << 42) / (((uint64_t)1 << 42) + ratio * coarse);
  uint64_t uCube = (((u * u) >> 22) * u) >> 22;
  uint64_t correction = ((u * kInverseTwoPiQ40) >> 7) - (((uCube * kInverseTwoPiQ40) / 3) >> 27);
  uint64_t total = ((uint64_t)arctangentTable[coarse] << 32) + correction;
  uint64_t base = total >> 33;
  uint64_t rest = total - (base << 33);
  if (rest > ((uint64_t)1 << 32) || (rest == ((uint64_t)1 << 32) && (base & 1)))
    ++base;
  int64_t folded = swapped ? ((int64_t)1 << 30) - (int64_t)base : (int64_t)base;
  int64_t turn;
  if (real >= 0)
    turn = imaginary >= 0 ? folded : -folded;
  else
    turn = imaginary >= 0 ? ((int64_t)1 << 31) - folded : ((int64_t)1 << 31) + folded;
  return (uint32_t)turn;
}

static uint32_t nearestTurn(int64_t real, int64_t imaginary) {
  const long double twoPi = 6.283185307179586476925286766559005768394L;
  long double turn = atan2l((long double)imaginary, (long double)real) / twoPi;
  if (turn < 0.0L)
    turn += 1.0L;
  return (uint32_t)(uint64_t)roundl(turn * 4294967296.0L);
}

static void report(const char *label, int64_t real, int64_t imaginary, uint32_t got,
                   uint32_t expected) {
  if (failures++ < 8)
    fprintf(stderr, "%s: phase(%lld, %lld) got %u, expected %u\n", label, (long long)real,
            (long long)imaginary, got, expected);
}

static void checkOne(const char *label, int64_t real, int64_t imaginary, uint32_t got) {
  uint32_t contract = referenceTurn(real, imaginary);
  if (got != contract)
    report(label, real, imaginary, got, contract);
  if (real == 0 && imaginary == 0)
    return;
  int64_t distance = (int64_t)(int32_t)(got - nearestTurn(real, imaginary));
  if (distance < 0)
    distance = -distance;
  if (distance > worstDistance)
    worstDistance = distance;
  if (distance > kMaxDistance)
    report("distance", real, imaginary, got, nearestTurn(real, imaginary));
}

static void checkNarrow(const int16_t *real, const int16_t *imaginary, const char *label) {
  int32_t packed[kBlock];
  for (int64_t i = 0; i < kBlock; ++i)
    packed[i] = (int32_t)(((uint32_t)(uint16_t)imaginary[i] << 16) | (uint16_t)real[i]);
  MemRefI32 inputRef = {packed, packed, 0, {kBlock}, {1}};
  MemRefI32 output;
  _mlir_ciface_phase_q15_turn32(&output, &inputRef);
  for (int64_t i = 0; i < kBlock; ++i)
    checkOne(label, real[i], imaginary[i], (uint32_t)output.aligned[output.offset + i]);
  free(output.allocated);
}

static void checkWide(const int32_t *real, const int32_t *imaginary, const char *label) {
  int64_t packed[kBlock];
  for (int64_t i = 0; i < kBlock; ++i)
    packed[i] = (int64_t)(((uint64_t)(uint32_t)imaginary[i] << 32) | (uint32_t)real[i]);
  MemRefI64 inputRef = {packed, packed, 0, {kBlock}, {1}};
  MemRefI32 output;
  _mlir_ciface_phase_q31_turn32(&output, &inputRef);
  for (int64_t i = 0; i < kBlock; ++i)
    checkOne(label, real[i], imaginary[i], (uint32_t)output.aligned[output.offset + i]);
  free(output.allocated);
}

int main(void) {
  buildTable();
  int16_t nr[kBlock], ni[kBlock];
  int32_t wr[kBlock], wi[kBlock];

  /* The angles the contract names exactly, at both widths: the axes give the
   * quarter turns and the diagonals the odd eighths, by exact arithmetic. */
  struct {
    int32_t real, imaginary;
    uint32_t turn;
  } named[] = {
      {0, 0, 0u},
      {1, 0, 0u},
      {0, 1, 1073741824u},
      {-1, 0, 2147483648u},
      {0, -1, 3221225472u},
      {1, 1, 536870912u},
      {-1, 1, 1610612736u},
      {-1, -1, 2684354560u},
      {1, -1, 3758096384u},
      {32767, 32767, 536870912u},
      {-32768, -32768, 2684354560u},
      {INT32_MAX, INT32_MAX, 536870912u},
      {INT32_MIN, INT32_MIN, 2684354560u},
      {INT32_MIN, 0, 2147483648u},
      {0, INT32_MIN, 3221225472u},
      {INT32_MAX, 0, 0u},
  };
  int64_t namedCount = (int64_t)(sizeof named / sizeof named[0]);
  for (int64_t i = 0; i < kBlock; ++i) {
    wr[i] = named[i % namedCount].real;
    wi[i] = named[i % namedCount].imaginary;
    nr[i] = (int16_t)(wr[i] > 32767 ? 32767 : wr[i] < -32768 ? -32768 : wr[i]);
    ni[i] = (int16_t)(wi[i] > 32767 ? 32767 : wi[i] < -32768 ? -32768 : wi[i]);
  }
  checkWide(wr, wi, "named wide");
  checkNarrow(nr, ni, "named narrow");
  {
    int64_t packed[kBlock];
    for (int64_t i = 0; i < kBlock; ++i)
      packed[i] = (int64_t)(((uint64_t)(uint32_t)wi[i] << 32) | (uint32_t)wr[i]);
    MemRefI64 inputRef = {packed, packed, 0, {kBlock}, {1}};
    MemRefI32 output;
    _mlir_ciface_phase_q31_turn32(&output, &inputRef);
    for (int64_t i = 0; i < namedCount; ++i)
      if ((uint32_t)output.aligned[output.offset + i] != named[i].turn)
        report("named turn", named[i].real, named[i].imaginary,
               (uint32_t)output.aligned[output.offset + i], named[i].turn);
    free(output.allocated);
  }

  /* Narrow components: one component swept whole against fixed pivots of
   * the other, the axes and both diagonals over their whole range. */
  static const int16_t pivots[] = {1, -1, 3, 181, 32767, -32768, 16384};
  for (size_t p = 0; p < sizeof pivots / sizeof pivots[0]; ++p)
    for (int64_t block = 0; block < 16; ++block) {
      for (int64_t i = 0; i < kBlock; ++i) {
        nr[i] = pivots[p];
        ni[i] = (int16_t)(int32_t)(block * kBlock + i - 32768);
      }
      checkNarrow(nr, ni, "narrow imaginary sweep");
      for (int64_t i = 0; i < kBlock; ++i) {
        nr[i] = (int16_t)(int32_t)(block * kBlock + i - 32768);
        ni[i] = pivots[p];
      }
      checkNarrow(nr, ni, "narrow real sweep");
    }
  for (int64_t block = 0; block < 16; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t v = (int16_t)(int32_t)(block * kBlock + i - 32768);
      nr[i] = v;
      ni[i] = (int16_t)((i & 1) ? -v : v);
    }
    checkNarrow(nr, ni, "narrow diagonals");
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t v = (int16_t)(int32_t)(block * kBlock + i - 32768);
      nr[i] = (i & 1) ? 0 : v;
      ni[i] = (i & 1) ? v : 0;
    }
    checkNarrow(nr, ni, "narrow axes");
  }

  /* Wide components: a stride walk of one component against full-scale and
   * unit pivots of the other, the wide axes and diagonals on the same walk,
   * and a deterministic random sweep of both. */
  uint32_t state = 0u;
  for (int64_t block = 0; block < 64; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      state += 1021u * 1048573u;
      int32_t moving = (int32_t)state;
      static const int32_t wide[] = {INT32_MAX, INT32_MIN, 1, -1, 1073741824, 3};
      wr[i] = (i & 1) ? moving : wide[(block + i / 2) % 6];
      wi[i] = (i & 1) ? wide[(block + i / 2) % 6] : moving;
    }
    checkWide(wr, wi, "wide pivot sweep");
    for (int64_t i = 0; i < kBlock; ++i) {
      state += 1021u * 1048573u;
      int32_t v = (int32_t)state;
      wr[i] = (i & 2) ? 0 : v;
      wi[i] = (i & 2) ? v : ((i & 1) ? (int32_t) - (int64_t)v : v);
      if ((i & 3) == 3)
        wr[i] = v;
    }
    checkWide(wr, wi, "wide axes and diagonals");
  }
  state = 0x6C1D93AFu;
  for (int64_t block = 0; block < 256; ++block) {
    for (int64_t i = 0; i < kBlock; ++i) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      wr[i] = (int32_t)state;
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      wi[i] = (int32_t)state;
    }
    checkWide(wr, wi, "wide random");
  }

  printf("cx_phase Q0.32: worst distance %lld LSB from the correctly rounded turn\n",
         (long long)worstDistance);
  return failures != 0;
}
