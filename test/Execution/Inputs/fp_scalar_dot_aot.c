#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern float dot_f32_off(float lhs, float rhs);
extern float dot_f32_fma(float lhs, float rhs);

static uint32_t float_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

int main(void) {
  const float negative_zero = -0.0f;
  const float one = 1.0f;
  uint32_t off_bits = float_bits(dot_f32_off(negative_zero, one));
  uint32_t fma_bits = float_bits(dot_f32_fma(negative_zero, one));

  if (off_bits != UINT32_C(0x80000000)) {
    fprintf(stderr, "off contract: expected negative zero, got 0x%08x\n", off_bits);
    return 1;
  }
  if (fma_bits != UINT32_C(0x00000000)) {
    fprintf(stderr, "fma contract: expected positive zero, got 0x%08x\n", fma_bits);
    return 1;
  }
  return 0;
}
