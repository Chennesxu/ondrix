// Independent reference for the Q31 packed complex reduction: exact __int128
// terms, an i64 accumulator clamped after every update, and one nearest-even
// export per component.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TERMS 8

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
} MemRef1D;

extern int64_t _mlir_ciface_cx_dot_q31(MemRef1D *lhs, MemRef1D *rhs);
extern int64_t _mlir_ciface_cx_corr_q31(MemRef1D *lhs, MemRef1D *rhs);

static int32_t component_low(uint64_t packed) { return (int32_t)(uint32_t)packed; }
static int32_t component_high(uint64_t packed) { return (int32_t)(uint32_t)(packed >> 32); }

static int64_t clamp64(__int128 value) {
  if (value > (__int128)INT64_MAX)
    return INT64_MAX;
  if (value < (__int128)INT64_MIN)
    return INT64_MIN;
  return (int64_t)value;
}

// frac 62 to frac 31 under nearest-even, saturated to the component storage.
static int32_t export_component(int64_t value) {
  const int shift = 31;
  int64_t quotient = value >> shift;
  uint64_t remainder = (uint64_t)value & ((1ULL << shift) - 1u);
  uint64_t half = 1ULL << (shift - 1);
  int64_t increment = 0;
  if (remainder > half)
    increment = 1;
  else if (remainder == half)
    increment = (quotient & 1) ? 1 : 0;
  int64_t rounded = quotient + increment;
  if (rounded > INT32_MAX)
    return INT32_MAX;
  if (rounded < INT32_MIN)
    return INT32_MIN;
  return (int32_t)rounded;
}

static uint64_t reference(const uint64_t *lhs, const uint64_t *rhs, int conjugate) {
  int64_t real = 0;
  int64_t imag = 0;
  for (int index = 0; index < TERMS; ++index) {
    // The layout puts the real component in the low half.
    __int128 xr = component_low(lhs[index]);
    __int128 xi = component_high(lhs[index]);
    __int128 yr = component_low(rhs[index]);
    __int128 yi = component_high(rhs[index]);
    __int128 term_real = conjugate ? xr * yr + xi * yi : xr * yr - xi * yi;
    __int128 term_imag = conjugate ? xi * yr - xr * yi : xr * yi + xi * yr;
    real = clamp64((__int128)real + term_real);
    imag = clamp64((__int128)imag + term_imag);
  }
  return ((uint64_t)(uint32_t)export_component(imag) << 32) |
         (uint64_t)(uint32_t)export_component(real);
}

// Rails first, then asymmetric mixtures: a symmetric case cannot separate the
// two conjugate readings, and without the rails the i64 clamp never fires.
static const uint64_t CASES[][2][TERMS] = {
    {{0x7FFFFFFF7FFFFFFFull, 0x7FFFFFFF7FFFFFFFull, 0x7FFFFFFF7FFFFFFFull, 0x7FFFFFFF7FFFFFFFull,
      0x7FFFFFFF7FFFFFFFull, 0x7FFFFFFF7FFFFFFFull, 0x7FFFFFFF7FFFFFFFull, 0x7FFFFFFF7FFFFFFFull},
     {0x7FFFFFFF80000000ull, 0x7FFFFFFF80000000ull, 0x7FFFFFFF80000000ull, 0x7FFFFFFF80000000ull,
      0x7FFFFFFF80000000ull, 0x7FFFFFFF80000000ull, 0x7FFFFFFF80000000ull, 0x7FFFFFFF80000000ull}},
    {{0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull,
      0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull},
     {0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull,
      0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull, 0x8000000080000000ull}},
    {{0x0000000100000002ull, 0xFFFFFFFF00000003ull, 0x000000047FFFFFFFull, 0x8000000000000005ull,
      0x0000000600000007ull, 0x123456789ABCDEF0ull, 0x0FEDCBA987654321ull, 0x00000000FFFFFFFFull},
     {0x00000003FFFFFFFFull, 0x7FFFFFFF00000001ull, 0x0000000280000000ull, 0xFFFFFFFF7FFFFFFFull,
      0x4000000040000000ull, 0xC0000000C0000000ull, 0x0000000000000001ull, 0x7FFFFFFF7FFFFFFFull}},
    {{0x4000000040000000ull, 0xC000000040000000ull, 0x40000000C0000000ull, 0xC0000000C0000000ull,
      0x2000000060000000ull, 0xA0000000E0000000ull, 0x0000000100000001ull, 0xFFFFFFFFFFFFFFFFull},
     {0x6000000020000000ull, 0x20000000E0000000ull, 0xE000000060000000ull, 0xA000000020000000ull,
      0x7FFFFFFF00000000ull, 0x000000007FFFFFFFull, 0x8000000000000001ull, 0x0000000180000000ull}},
};

#define CASE_COUNT ((int)(sizeof(CASES) / sizeof(CASES[0])))

static MemRef1D descriptor(uint64_t *storage) {
  MemRef1D descriptor;
  descriptor.allocated = (int64_t *)storage;
  descriptor.aligned = (int64_t *)storage;
  descriptor.offset = 0;
  descriptor.size = TERMS;
  descriptor.stride = 1;
  return descriptor;
}

int main(void) {
  int failures = 0;
  int clamped = 0;
  int separated = 0;
  for (int index = 0; index < CASE_COUNT; ++index) {
    uint64_t lhs[TERMS];
    uint64_t rhs[TERMS];
    memcpy(lhs, CASES[index][0], sizeof(lhs));
    memcpy(rhs, CASES[index][1], sizeof(rhs));
    MemRef1D lhsDescriptor = descriptor(lhs);
    MemRef1D rhsDescriptor = descriptor(rhs);

    uint64_t wantDot = reference(lhs, rhs, 0);
    uint64_t wantCorr = reference(lhs, rhs, 1);
    if (wantDot != wantCorr)
      ++separated;
    if (component_low(wantDot) == INT32_MAX || component_low(wantDot) == INT32_MIN ||
        component_high(wantDot) == INT32_MAX || component_high(wantDot) == INT32_MIN)
      ++clamped;

    uint64_t gotDot = (uint64_t)_mlir_ciface_cx_dot_q31(&lhsDescriptor, &rhsDescriptor);
    uint64_t gotCorr = (uint64_t)_mlir_ciface_cx_corr_q31(&lhsDescriptor, &rhsDescriptor);
    if (gotDot != wantDot) {
      printf("case %d dot: got %016llx want %016llx\n", index, (unsigned long long)gotDot,
             (unsigned long long)wantDot);
      ++failures;
    }
    if (gotCorr != wantCorr) {
      printf("case %d correlate: got %016llx want %016llx\n", index, (unsigned long long)gotCorr,
             (unsigned long long)wantCorr);
      ++failures;
    }
    // A kernel must not write through its read-only operands.
    if (memcmp(lhs, CASES[index][0], sizeof(lhs)) != 0 ||
        memcmp(rhs, CASES[index][1], sizeof(rhs)) != 0) {
      printf("case %d mutated an input buffer\n", index);
      ++failures;
    }
  }
  if (separated < CASE_COUNT - 1) {
    printf("corpus separates the conjugate reading on only %d of %d cases\n", separated,
           CASE_COUNT);
    ++failures;
  }
  if (clamped == 0) {
    printf("corpus never reaches the export rail\n");
    ++failures;
  }
  if (failures != 0) {
    printf("cx_dot_q31: %d failures\n", failures);
    return 1;
  }
  printf("cx_dot_q31: %d cases, %d reaching the rail, all match the reference\n", CASE_COUNT,
         clamped);
  return 0;
}
