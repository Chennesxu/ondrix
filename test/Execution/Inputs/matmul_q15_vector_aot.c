#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRefI16x2;

extern void _mlir_ciface_matmul8x8x8_q15_vector(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);
extern void _mlir_ciface_matmul4x16x3_q15_vector(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);
extern void _mlir_ciface_matmul1x64x1_q15_vector(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);
extern void _mlir_ciface_matmul2x12x2_q15_vector(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);

enum { kMaxDim = 64, kTrialCount = 12 };

/* Independent contract arithmetic: exact i64 K-sum, one round-half-even
 * shift by 15 in explicit floor-division form, i16 saturation. The compiled
 * kernel reaches the same values through a transposed pack, a horizontal
 * Vector reduction over an i40 wrapping accumulator, and one acc_export. */
static int16_t referenceElement(const int16_t *a, const int16_t *b, int64_t inner, int64_t columns,
                                int64_t row, int64_t column) {
  int64_t sum = 0;
  for (int64_t k = 0; k < inner; ++k)
    sum += (int64_t)a[row * inner + k] * b[k * columns + column];
  int64_t quotient = sum / 32768;
  int64_t remainder = sum % 32768;
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

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

static int16_t toSigned16(uint32_t bits) {
  uint32_t low = bits & 0xFFFFu;
  return (int16_t)(low < 32768u ? (int32_t)low : (int32_t)low - 65536);
}

static int check(void (*kernel)(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *), const int16_t *a,
                 const int16_t *b, int64_t rows, int64_t inner, int64_t columns,
                 const char *label) {
  MemRefI16x2 lhs = {(int16_t *)a, (int16_t *)a, 0, {rows, inner}, {inner, 1}};
  MemRefI16x2 rhs = {(int16_t *)b, (int16_t *)b, 0, {inner, columns}, {columns, 1}};
  MemRefI16x2 output;
  kernel(&output, &lhs, &rhs);

  int failed = output.sizes[0] != rows || output.sizes[1] != columns;
  for (int64_t i = 0; !failed && i < rows; ++i)
    for (int64_t j = 0; j < columns; ++j) {
      int16_t expected = referenceElement(a, b, inner, columns, i, j);
      int16_t actual =
          output.aligned[output.offset + i * output.strides[0] + j * output.strides[1]];
      if (actual != expected) {
        fprintf(stderr, "%s [%lld][%lld]: got %d, expected %d\n", label, (long long)i, (long long)j,
                actual, expected);
        failed = 1;
      }
    }
  free(output.allocated);
  return failed;
}

int main(void) {
  int failed = 0;
  static int16_t a[kMaxDim * kMaxDim];
  static int16_t b[kMaxDim * kMaxDim];

  /* Q15 "identity": 32767 on the diagonal. This is 32767/32768 scaling,
   * not an exact identity; the reference decides every element. */
  for (int64_t i = 0; i < 64; ++i) {
    a[i] = 0;
    b[i] = 0;
  }
  for (int64_t i = 0; i < 8; ++i)
    b[i * 8 + i] = INT16_MAX;
  uint32_t state = 0x5C1D0B37u;
  for (int64_t i = 0; i < 64; ++i) {
    state = nextState(state);
    a[i] = toSigned16(state);
  }
  failed |= check(_mlir_ciface_matmul8x8x8_q15_vector, a, b, 8, 8, 8, "near identity");

  /* Every output saturates: full rows of -32768 against -32768 columns
   * give K * 2^30 sums, far above the Q15 ceiling. */
  for (int64_t i = 0; i < 64; ++i) {
    a[i] = INT16_MIN;
    b[i] = INT16_MIN;
  }
  failed |= check(_mlir_ciface_matmul8x8x8_q15_vector, a, b, 8, 8, 8, "all saturate");

  /* Longest admitted reduction at the extreme magnitude: 64 * 2^30 = 2^36,
   * the bound that keeps the i40 wrapping accumulator from wrapping. */
  for (int64_t i = 0; i < 64; ++i) {
    a[i] = INT16_MIN;
    b[i] = INT16_MIN;
  }
  failed |= check(_mlir_ciface_matmul1x64x1_q15_vector, a, b, 1, 64, 1, "max inner saturate");
  for (int64_t i = 0; i < 64; ++i)
    b[i] = INT16_MAX;
  failed |= check(_mlir_ciface_matmul1x64x1_q15_vector, a, b, 1, 64, 1, "max inner opposite");

  /* K = 12 splits into one eight-lane horizontal chunk and a four-step
   * ordered scalar tail, so the mixed schedule is executed rather than only
   * compiled. The all-minimum corner sums 12 * 2^30 and saturates; the
   * min-times-max corner is the largest negative sum, -12 * 32768 * 32767. */
  for (int64_t i = 0; i < 24; ++i) {
    a[i] = INT16_MIN;
    b[i] = INT16_MIN;
  }
  failed |= check(_mlir_ciface_matmul2x12x2_q15_vector, a, b, 2, 12, 2, "mixed schedule saturate");
  for (int64_t i = 0; i < 24; ++i)
    b[i] = INT16_MAX;
  failed |= check(_mlir_ciface_matmul2x12x2_q15_vector, a, b, 2, 12, 2, "mixed schedule opposite");

  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    for (int64_t i = 0; i < kMaxDim * kMaxDim; ++i) {
      state = nextState(state);
      a[i] = toSigned16(state);
    }
    for (int64_t i = 0; i < kMaxDim * kMaxDim; ++i) {
      state = nextState(state);
      b[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "square trial %d", trial);
    failed |= check(_mlir_ciface_matmul8x8x8_q15_vector, a, b, 8, 8, 8, label);
    snprintf(label, sizeof label, "rect trial %d", trial);
    failed |= check(_mlir_ciface_matmul4x16x3_q15_vector, a, b, 4, 16, 3, label);
    snprintf(label, sizeof label, "inner trial %d", trial);
    failed |= check(_mlir_ciface_matmul1x64x1_q15_vector, a, b, 1, 64, 1, label);
    snprintf(label, sizeof label, "mixed schedule trial %d", trial);
    failed |= check(_mlir_ciface_matmul2x12x2_q15_vector, a, b, 2, 12, 2, label);
  }
  return failed;
}
