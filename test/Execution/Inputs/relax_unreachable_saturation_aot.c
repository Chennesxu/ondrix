#include <stdint.h>
#include <stdio.h>

struct MemRef1D {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

extern void _mlir_ciface_dct8_q15(struct MemRef1D *out, struct MemRef1D *in);

// The declared output format leaves the reading a bit of room, so the rails
// are exactly where a dropped clamp would show if the proof were wrong.
static void fill(int16_t *data, int pattern, uint32_t *state) {
  for (int i = 0; i < 8; ++i) {
    if (pattern == 0)
      data[i] = -32768;
    else if (pattern == 1)
      data[i] = 32767;
    else if (pattern == 2)
      data[i] = (i & 1) ? 32767 : -32768;
    else if (pattern == 3)
      data[i] = (i & 1) ? -32768 : 32767;
    else {
      *state = *state * 1103515245u + 12345u;
      data[i] = (int16_t)(*state >> 16);
    }
  }
}

int main(void) {
  int16_t input[8], buffer[8];
  uint32_t state = 1u;
  for (int pattern = 0; pattern < 400; ++pattern) {
    fill(input, pattern, &state);
    struct MemRef1D source = {input, input, 0, 8, 1};
    struct MemRef1D result = {buffer, buffer, 0, 8, 1};
    _mlir_ciface_dct8_q15(&result, &source);
    for (int i = 0; i < 8; ++i)
      printf("%d %d %d\n", pattern, i, (int)result.aligned[result.offset + i]);
  }
  return 0;
}
