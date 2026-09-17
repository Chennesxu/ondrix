#include <stdint.h>
#include <stdio.h>

struct MemRef1D {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

extern void _mlir_ciface_dct32_q15(struct MemRef1D *, struct MemRef1D *);

// The rails first: a column read from memory and a column materialized as an
// immediate must agree everywhere, and the rails are where a transposed or
// misaligned table would show.
static uint32_t state = 1u;

int main(void) {
  int16_t input[32], output[32];
  for (int pattern = 0; pattern < 300; ++pattern) {
    for (int i = 0; i < 32; ++i) {
      if (pattern == 0)
        input[i] = -32768;
      else if (pattern == 1)
        input[i] = 32767;
      else if (pattern == 2)
        input[i] = (i & 1) ? 32767 : -32768;
      else {
        state = state * 1103515245u + 12345u;
        input[i] = (int16_t)(state >> 16);
      }
    }
    struct MemRef1D source = {input, input, 0, 32, 1};
    struct MemRef1D result = {output, output, 0, 32, 1};
    _mlir_ciface_dct32_q15(&result, &source);
    for (int i = 0; i < 32; ++i)
      printf("%d %d %d\n", pattern, i, (int)result.aligned[result.offset + i]);
  }
  return 0;
}
