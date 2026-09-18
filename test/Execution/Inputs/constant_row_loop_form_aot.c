#include <stdint.h>
#include <stdio.h>

// The canonical pipeline gives a static tensor result an OUT PARAMETER, so the
// generated entry point takes the input first and the caller's buffer last.
extern void ondrix_dct32_q15(const int16_t *input, int16_t *output);

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
      // A sentinel no rail can produce: an output the kernel never wrote would
      // otherwise let two schedules agree on uninitialized memory.
      output[i] = 12345;
    }
    ondrix_dct32_q15(input, output);
    int written = 0;
    for (int i = 0; i < 32; ++i)
      written |= output[i] != 12345;
    if (!written) {
      fprintf(stderr, "pattern %d: the kernel wrote no output\n", pattern);
      return 2;
    }
    for (int i = 0; i < 32; ++i)
      printf("%d %d %d\n", pattern, i, (int)output[i]);
  }
  return 0;
}
