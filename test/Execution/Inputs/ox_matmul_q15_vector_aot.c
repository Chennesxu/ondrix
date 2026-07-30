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

extern void _mlir_ciface_q15_matmul(MemRefI16x2 *, MemRefI16x2 *, MemRefI16x2 *);

enum { kRows = 4, kInner = 8, kColumns = 3, kTrialCount = 12 };

/* Independent contract arithmetic: exact i64 K-sum, one round-half-even
 * shift by 15 in explicit floor-division form, i16 saturation. Identical to
 * the reference that pins matmul_q15_vector_aot.mlir; here it decides the
 * frontend-compiled kernel. */
static int16_t referenceElement(const int16_t *a, const int16_t *b, int64_t row, int64_t column) {
  int64_t sum = 0;
  for (int64_t k = 0; k < kInner; ++k)
    sum += (int64_t)a[row * kInner + k] * b[k * kColumns + column];
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

static int check(const int16_t *a, const int16_t *b, const char *label) {
  MemRefI16x2 lhs = {(int16_t *)a, (int16_t *)a, 0, {kRows, kInner}, {kInner, 1}};
  MemRefI16x2 rhs = {(int16_t *)b, (int16_t *)b, 0, {kInner, kColumns}, {kColumns, 1}};
  MemRefI16x2 output;
  _mlir_ciface_q15_matmul(&output, &lhs, &rhs);

  int failed = output.sizes[0] != kRows || output.sizes[1] != kColumns;
  for (int64_t i = 0; !failed && i < kRows; ++i)
    for (int64_t j = 0; j < kColumns; ++j) {
      int16_t expected = referenceElement(a, b, i, j);
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
  int16_t a[kRows * kInner];
  int16_t b[kInner * kColumns];

  for (int64_t i = 0; i < kRows * kInner; ++i)
    a[i] = 0;
  for (int64_t i = 0; i < kInner * kColumns; ++i)
    b[i] = 0;
  failed |= check(a, b, "zero");

  /* Every output saturates: full rows of -32768 against -32768 columns give
   * K * 2^30 = 2^33 sums, far above the Q15 ceiling. */
  for (int64_t i = 0; i < kRows * kInner; ++i)
    a[i] = INT16_MIN;
  for (int64_t i = 0; i < kInner * kColumns; ++i)
    b[i] = INT16_MIN;
  failed |= check(a, b, "all saturate");

  /* The largest negative sums: -8 * 32768 * 32767 per element. */
  for (int64_t i = 0; i < kInner * kColumns; ++i)
    b[i] = INT16_MAX;
  failed |= check(a, b, "opposite rails");

  /* A single nonzero product per output rides on one lane of the chunk. */
  for (int64_t i = 0; i < kRows * kInner; ++i)
    a[i] = 0;
  a[0] = INT16_MAX;
  failed |= check(a, b, "single impulse");

  uint32_t state = 0x3A9E17C5u;
  for (int trial = 0; trial < kTrialCount; ++trial) {
    char label[32];
    for (int64_t i = 0; i < kRows * kInner; ++i) {
      state = nextState(state);
      a[i] = toSigned16(state);
    }
    for (int64_t i = 0; i < kInner * kColumns; ++i) {
      state = nextState(state);
      b[i] = toSigned16(state);
    }
    snprintf(label, sizeof label, "trial %d", trial);
    failed |= check(a, b, label);
  }
  return failed;
}
