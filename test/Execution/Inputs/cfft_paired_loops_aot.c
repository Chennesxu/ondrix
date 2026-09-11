#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_cfft32_floor_wrap(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_icfft32_ntp_sat(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_cfft64_ntp_sat(MemRefI32 *, MemRefI32 *);
extern void _mlir_ciface_icfft64_floor_wrap(MemRefI32 *, MemRefI32 *);

enum { kTrialCount = 6 };

static uint32_t nextState(uint32_t state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

/* Prints every output word; the lit test diffs the prints of the routes. */
static void dump(const char *label, void (*kernel)(MemRefI32 *, MemRefI32 *), unsigned extent,
                 uint32_t *state) {
  int32_t *input = malloc(extent * sizeof *input);
  for (int trial = 0; trial < kTrialCount; ++trial) {
    for (unsigned i = 0; i < extent; ++i) {
      *state = nextState(*state);
      /* Trial 0 rails every component; later trials walk the random field. */
      input[i] =
          trial == 0
              ? (int32_t)(((i & 1) ? 0x8000u : 0x7FFFu) | ((i & 2) ? 0x7FFF0000u : 0x80000000u))
              : (int32_t)*state;
    }
    MemRefI32 inputRef = {input, input, 0, {extent}, {1}};
    MemRefI32 output;
    kernel(&output, &inputRef);
    printf("%s trial %d size %lld\n", label, trial, (long long)output.sizes[0]);
    for (int64_t i = 0; i < output.sizes[0]; ++i)
      printf("%08x\n", (unsigned)output.aligned[output.offset + i * output.strides[0]]);
    free(output.allocated);
  }
  free(input);
}

int main(void) {
  uint32_t state = 0x9E3779B9u;
  dump("cfft32_floor_wrap", _mlir_ciface_cfft32_floor_wrap, 32, &state);
  dump("icfft32_ntp_sat", _mlir_ciface_icfft32_ntp_sat, 32, &state);
  dump("cfft64_ntp_sat", _mlir_ciface_cfft64_ntp_sat, 64, &state);
  dump("icfft64_floor_wrap", _mlir_ciface_icfft64_floor_wrap, 64, &state);
  return 0;
}
