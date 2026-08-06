#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern void ortumcore_fft8(int16_t *, int16_t *, int64_t, int64_t, int64_t, int16_t *, int16_t *,
                           int64_t, int64_t, int64_t, int16_t *, int16_t *, int64_t, int64_t,
                           int64_t, int16_t *, int16_t *, int64_t, int64_t, int64_t);

#define MEMREF_ARGS(pointer, size) pointer, pointer, INT64_C(0), INT64_C(size), INT64_C(1)

static int16_t twiddle_re[4] = {32767, 23170, 0, -23170};
static int16_t twiddle_im[4] = {0, -23170, -32768, -23170};

// Independent formulation: floor as explicit division, the permutation as a
// digit walk — neither matches the emulation lowering's shifts or the
// kernel's bitrev_add walk.
static int32_t floor_shift(int64_t value, unsigned shift) {
  int64_t divisor = (int64_t)1 << shift;
  int64_t quotient = value / divisor;
  if (value % divisor != 0 && value < 0)
    --quotient;
  return (int32_t)quotient;
}

static unsigned reverse3(unsigned index) {
  unsigned reversed = 0;
  for (unsigned bit = 0; bit < 3; ++bit)
    reversed |= ((index >> bit) & 1u) << (2 - bit);
  return reversed;
}

static int16_t narrow(int32_t value, int *range_failed) {
  if (value < INT16_MIN || value > INT16_MAX)
    *range_failed = 1;
  return (int16_t)value;
}

static void fft8_reference(int16_t *re, int16_t *im, int truncate, int reorder, int *range_failed) {
  if (reorder)
    for (unsigned k = 0; k < 8; ++k) {
      unsigned j = reverse3(k);
      if (j > k) {
        int16_t r = re[k], i = im[k];
        re[k] = re[j];
        im[k] = im[j];
        re[j] = r;
        im[j] = i;
      }
    }
  for (unsigned s = 0; s < 3; ++s) {
    unsigned half = 1u << s;
    unsigned stride = 4u >> s;
    for (unsigned base = 0; base < 8; base += 2 * half)
      for (unsigned j = 0; j < half; ++j) {
        int64_t wr = twiddle_re[j * stride], wi = twiddle_im[j * stride];
        unsigned a = base + j, b = a + half;
        int64_t pr = wr * re[b] - wi * im[b];
        int64_t pi = wr * im[b] + wi * re[b];
        int32_t tre = truncate ? (int32_t)(pr / 32768) : floor_shift(pr, 15);
        int32_t tim = truncate ? (int32_t)(pi / 32768) : floor_shift(pi, 15);
        int32_t xr = re[a], xi = im[a];
        re[a] = narrow(floor_shift(xr + tre, 1), range_failed);
        im[a] = narrow(floor_shift(xi + tim, 1), range_failed);
        re[b] = narrow(floor_shift(xr - tre, 1), range_failed);
        im[b] = narrow(floor_shift(xi - tim, 1), range_failed);
      }
  }
}

struct Vector {
  const char *name;
  int16_t re[8];
  int16_t im[8];
};

static const struct Vector vectors[] = {
    {"impulse", {16384, 0, 0, 0, 0, 0, 0, 0}, {0}},
    {"dc", {16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384}, {0}},
    {"tone bin 1",
     {20000, 14142, 0, -14142, -20000, -14142, 0, 14142},
     {0, 14142, 20000, 14142, 0, -14142, -20000, -14142}},
    {"mixed",
     {12345, -23169, 31000, -7, 0, 15000, -27001, 3333},
     {-6789, 9241, -55, 31999, -1, -15001, 2, -30303}},
};

static int check_vector(const struct Vector *vector) {
  int16_t kernel_re[8], kernel_im[8], reference_re[8], reference_im[8];
  memcpy(kernel_re, vector->re, sizeof(kernel_re));
  memcpy(kernel_im, vector->im, sizeof(kernel_im));
  memcpy(reference_re, vector->re, sizeof(reference_re));
  memcpy(reference_im, vector->im, sizeof(reference_im));

  int range_failed = 0;
  fft8_reference(reference_re, reference_im, 0, 1, &range_failed);
  if (range_failed) {
    fprintf(stderr, "%s: input leaves the no-saturation contract\n", vector->name);
    return 1;
  }
  ortumcore_fft8(MEMREF_ARGS(kernel_re, 8), MEMREF_ARGS(kernel_im, 8), MEMREF_ARGS(twiddle_re, 4),
                 MEMREF_ARGS(twiddle_im, 4));
  for (unsigned k = 0; k < 8; ++k)
    if (kernel_re[k] != reference_re[k] || kernel_im[k] != reference_im[k]) {
      fprintf(stderr, "%s bin %u: kernel (%d,%d), reference (%d,%d)\n", vector->name, k,
              kernel_re[k], kernel_im[k], reference_re[k], reference_im[k]);
      return 1;
    }
  return 0;
}

// Closed-form spectrum guards on the reference itself, so a reference bug
// cannot silently agree with a kernel bug.
static int check_spectrum(void) {
  int failed = 0, range_failed = 0;
  int16_t re[8], im[8];

  memcpy(re, vectors[0].re, sizeof(re));
  memcpy(im, vectors[0].im, sizeof(im));
  fft8_reference(re, im, 0, 1, &range_failed);
  for (unsigned k = 0; k < 8; ++k)
    if (re[k] != 2048 || im[k] != 0) {
      fprintf(stderr, "impulse bin %u: (%d,%d), expected (2048,0)\n", k, re[k], im[k]);
      failed = 1;
    }

  memcpy(re, vectors[2].re, sizeof(re));
  memcpy(im, vectors[2].im, sizeof(im));
  fft8_reference(re, im, 0, 1, &range_failed);
  for (unsigned k = 0; k < 8; ++k) {
    int32_t expected_re = k == 1 ? 20000 : 0;
    if (re[k] < expected_re - 8 || re[k] > expected_re + 8 || im[k] < -8 || im[k] > 8) {
      fprintf(stderr, "tone bin %u: (%d,%d) too far from (%d,0)\n", k, re[k], im[k], expected_re);
      failed = 1;
    }
  }
  return failed | range_failed;
}

// The mixed vector must discriminate both floor rounding and the reorder.
static int check_discrimination(void) {
  int failed = 0, range_failed = 0;
  int16_t floor_re[8], floor_im[8], other_re[8], other_im[8];

  memcpy(floor_re, vectors[3].re, sizeof(floor_re));
  memcpy(floor_im, vectors[3].im, sizeof(floor_im));
  fft8_reference(floor_re, floor_im, 0, 1, &range_failed);

  memcpy(other_re, vectors[3].re, sizeof(other_re));
  memcpy(other_im, vectors[3].im, sizeof(other_im));
  fft8_reference(other_re, other_im, 1, 1, &range_failed);
  if (memcmp(floor_re, other_re, sizeof(floor_re)) == 0 &&
      memcmp(floor_im, other_im, sizeof(floor_im)) == 0) {
    fprintf(stderr, "mixed vector cannot discriminate floor from truncation\n");
    failed = 1;
  }

  memcpy(other_re, vectors[3].re, sizeof(other_re));
  memcpy(other_im, vectors[3].im, sizeof(other_im));
  fft8_reference(other_re, other_im, 0, 0, &range_failed);
  if (memcmp(floor_re, other_re, sizeof(floor_re)) == 0 &&
      memcmp(floor_im, other_im, sizeof(floor_im)) == 0) {
    fprintf(stderr, "mixed vector cannot discriminate the reorder\n");
    failed = 1;
  }
  return failed;
}

int main(void) {
  int failed = check_spectrum() | check_discrimination();
  for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i)
    failed |= check_vector(&vectors[i]);
  return failed;
}
