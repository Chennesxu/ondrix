#include <stdint.h>
#include <stdio.h>

struct MemRef1D {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

extern int16_t _mlir_ciface_dot_pairs(struct MemRef1D *x);
extern int16_t _mlir_ciface_dot_reversed(struct MemRef1D *x);
extern int16_t _mlir_ciface_dot_long(struct MemRef1D *x);

static struct MemRef1D view(int16_t *data, int64_t size) {
  struct MemRef1D m = {data, data, 0, size, 1};
  return m;
}

// Rails first, then a pseudo-random sweep: the certificate's domain is the
// whole signed range, so the rails are where the two expansions could part.
static void fill(int16_t *data, int64_t size, int pattern, uint32_t *state) {
  for (int64_t i = 0; i < size; ++i) {
    if (pattern == 0)
      data[i] = -32768;
    else if (pattern == 1)
      data[i] = 32767;
    else if (pattern == 2)
      data[i] = (i & 1) ? 32767 : -32768;
    else {
      *state = *state * 1664525u + 1013904223u;
      data[i] = (int16_t)(*state >> 16);
    }
  }
}

int main(void) {
  int16_t eight[8], seventy[70];
  uint32_t state = 0x5EED1234u;
  for (int pattern = 0; pattern < 3 + 200; ++pattern) {
    fill(eight, 8, pattern, &state);
    fill(seventy, 70, pattern, &state);
    struct MemRef1D x8 = view(eight, 8), x70 = view(seventy, 70);
    printf("%d %d %d %d\n", pattern, _mlir_ciface_dot_pairs(&x8), _mlir_ciface_dot_reversed(&x8),
           _mlir_ciface_dot_long(&x70));
  }
  return 0;
}
