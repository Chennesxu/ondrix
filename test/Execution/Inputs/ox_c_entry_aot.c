#include "fixed_point_reference.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

// The descriptor entry the plain-pointer entry is held against.
extern int16_t q15_dot(int16_t *lhs_allocated, int16_t *lhs_aligned, int64_t lhs_offset,
                       int64_t lhs_size, int64_t lhs_stride, int16_t *rhs_allocated,
                       int16_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                       int64_t rhs_stride);

static const struct Policy q15_saturating = {
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

static int16_t reference(const int16_t *lhs, const int16_t *rhs, int64_t count,
                         const struct Policy *policy) {
  int64_t accumulator = 0;
  for (int64_t i = 0; i < count; ++i)
    accumulator = update_reference(accumulator, lhs[i], rhs[i], policy);
  return (int16_t)export_reference(accumulator, policy->output_rounding, policy->output_overflow,
                                   policy);
}

// The descriptor entries the tensor-result plain entries are held against.
typedef struct {
  int16_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRefI16;
typedef struct {
  int16_t *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
} MemRefI16x2;
extern void _mlir_ciface_q15_multi_use_binding(MemRefI16 *, MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_q15_matmul_floor(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);

static int16_t saturate16(int32_t value) {
  return (int16_t)(value > 32767 ? 32767 : (value < -32768 ? -32768 : value));
}

static int16_t roundShiftHalfEven(int32_t value, int shift) {
  int32_t quotient = value >> shift;
  int32_t remainder = value - (quotient << shift);
  int32_t half = (int32_t)1 << (shift - 1);
  if (remainder > half || (remainder == half && (quotient & 1)))
    ++quotient;
  return saturate16(quotient);
}

// t = mult(x, y); add(shift(mult(t, t), -1), t), each step nearest-even and saturating.
static int16_t chainReference(int16_t x, int16_t y) {
  int16_t t = roundShiftHalfEven((int32_t)x * y, 15);
  int16_t square = roundShiftHalfEven((int32_t)t * t, 15);
  return saturate16((int32_t)roundShiftHalfEven(square, 1) + t);
}

static int16_t nextSample(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return (int16_t)((int32_t)(*state % 65536u) - 32768);
}

// A refused length must abort before any element is read: the buffers are null.
static int refuses(uint64_t length) {
  pid_t child = fork();
  if (child == 0) {
    freopen("/dev/null", "w", stdout);
    ondrix_q15_dot(NULL, NULL, length);
    _exit(0);
  }
  int status = 0;
  if (child < 0 || waitpid(child, &status, 0) != child)
    return 0;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

int main(void) {
  int16_t lhs[] = {-32768, 32767, 16384, -16384, 1, -1, 12345, -23456, 32767};
  int16_t rhs[] = {-32768, 32767, 16384, 16384, 16384, 16384, -22222, 11111, -32768};
  uint64_t lengths[] = {0, 1, 2, 3, 8, 9};
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    int64_t length = (int64_t)lengths[i];
    int16_t entry = ondrix_q15_dot(lhs, rhs, lengths[i]);
    int16_t direct = q15_dot(lhs, lhs, 0, length, 1, rhs, rhs, 0, length, 1);
    int16_t expected = reference(lhs, rhs, length, &q15_saturating);
    if (entry != direct || entry != expected) {
      fprintf(stderr, "q15 dot length %lld: entry %d, descriptor %d, expected %d\n",
              (long long)length, entry, direct, expected);
      return 1;
    }
  }

  static const int16_t coefficients[5] = {16384, -8192, 4096, -8192, 16384};
  struct Policy wrapping = q15_saturating;
  wrapping.update_overflow = WRAP;
  int16_t windows[][5] = {
      {0, 0, 0, 0, 0},
      {-32768, 32767, 16384, -16384, 1},
      {32767, 32767, 32767, 32767, 32767},
      {-12345, 23456, -1, 1, 30000},
  };
  for (unsigned i = 0; i < sizeof(windows) / sizeof(windows[0]); ++i) {
    int16_t entry = ondrix_q15_fir_constexpr(windows[i]);
    int16_t expected = reference(windows[i], coefficients, 5, &wrapping);
    if (entry != expected) {
      fprintf(stderr, "constexpr q15 fir window %u: entry %d, expected %d\n", i, entry, expected);
      return 1;
    }
  }

  // A tensor result lands in the caller's array: entry, descriptor call and
  // reference agree, with the product that leaves Q1.15 in the corpus.
  int16_t x[32], y[32], chain[32], chainDirect[32];
  uint32_t state = 0x2F6E19A3u;
  for (int i = 0; i < 32; ++i) {
    x[i] = nextSample(&state);
    y[i] = nextSample(&state);
  }
  x[0] = -32768;
  y[0] = -32768;
  ondrix_q15_multi_use_binding(x, y, chain);
  MemRefI16 xRef = {x, x, 0, {32}, {1}};
  MemRefI16 yRef = {y, y, 0, {32}, {1}};
  MemRefI16 chainRef = {chainDirect, chainDirect, 0, {32}, {1}};
  _mlir_ciface_q15_multi_use_binding(&xRef, &yRef, &chainRef);
  for (int i = 0; i < 32; ++i) {
    int16_t expected = chainReference(x[i], y[i]);
    if (chain[i] != expected || chainDirect[i] != expected) {
      fprintf(stderr, "q15 chain[%d]: entry %d, descriptor %d, expected %d\n", i, chain[i],
              chainDirect[i], expected);
      return 1;
    }
  }

  // Rank 2 through the entry: only the pointer order is observable at run
  // time, the static sizes and strides are pinned by FileCheck.
  int16_t a[4][8], b[8][3], product[4][3], productDirect[4][3];
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 8; ++k)
      a[i][k] = nextSample(&state);
  for (int k = 0; k < 8; ++k)
    for (int j = 0; j < 3; ++j)
      b[k][j] = nextSample(&state);
  ondrix_q15_matmul_floor(&a[0][0], &b[0][0], &product[0][0]);
  MemRefI16x2 aRef = {&a[0][0], &a[0][0], 0, {4, 8}, {8, 1}};
  MemRefI16x2 bRef = {&b[0][0], &b[0][0], 0, {8, 3}, {3, 1}};
  MemRefI16x2 productRef = {&productDirect[0][0], &productDirect[0][0], 0, {4, 3}, {3, 1}};
  _mlir_ciface_q15_matmul_floor(&aRef, &bRef, &productRef);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      if (product[i][j] != productDirect[i][j]) {
        fprintf(stderr, "q15 matmul[%d][%d]: entry %d, descriptor %d\n", i, j, product[i][j],
                productDirect[i][j]);
        return 1;
      }

  if (!refuses(UINT64_MAX) || !refuses((uint64_t)1 << 63)) {
    fprintf(stderr, "a length past the signed index range was not refused\n");
    return 1;
  }
  return 0;
}
