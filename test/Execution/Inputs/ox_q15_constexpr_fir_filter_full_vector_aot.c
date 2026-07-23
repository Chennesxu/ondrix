#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI16;

extern void _mlir_ciface_q15_fir_filter_full_constexpr(MemRefI16 *, MemRefI16 *);

static int16_t reference(const int16_t *input, int64_t output) {
  static const int16_t coefficients[] = {1024, -512, 256, 128};
  int64_t accumulator = 0;
  for (int64_t tap = 0; tap < 4; ++tap) {
    int64_t inputIndex = output + tap - 3;
    if (inputIndex >= 0 && inputIndex < 8)
      accumulator += (int64_t)input[inputIndex] * coefficients[tap];
  }

  int64_t quotient = accumulator / 32768;
  int64_t remainder = accumulator % 32768;
  if (remainder < 0) {
    remainder += 32768;
    --quotient;
  }
  if (remainder > 16384 || (remainder == 16384 && (quotient & 1)))
    ++quotient;
  if (quotient < INT16_MIN)
    return INT16_MIN;
  if (quotient > INT16_MAX)
    return INT16_MAX;
  return (int16_t)quotient;
}

int main(void) {
  int16_t input[] = {32767, -32768, 16384, -8192, 4096, -2048, 1024, -512};
  MemRefI16 inputRef = {input, input, 0, {8}, {1}};
  MemRefI16 output;
  _mlir_ciface_q15_fir_filter_full_constexpr(&output, &inputRef);

  int failed = output.sizes[0] != 11;
  for (int64_t i = 0; i < output.sizes[0]; ++i) {
    int16_t expected = reference(input, i);
    int16_t actual = output.aligned[output.offset + i * output.strides[0]];
    if (actual != expected) {
      fprintf(stderr, "output %lld: got %d, expected %d\n", (long long)i, actual, expected);
      failed = 1;
    }
  }
  free(output.allocated);
  return failed;
}
