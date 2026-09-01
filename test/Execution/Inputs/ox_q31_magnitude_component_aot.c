#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Source-to-object gate for the Q31 magnitude component boundary. The two
 * objects come from two .ox kernels that differ only in `input_rounding`, and
 * the directed witness below forces them apart: re = -1 pre-shifts to 0 under
 * nearest_even (a tie, to even) and to -1 under toward_negative. */

typedef struct {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI64;

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRefI32;

extern void _mlir_ciface_q31_magnitude_component_even(MemRefI32 *, MemRefI64 *);
extern void _mlir_ciface_q31_magnitude_component_floor(MemRefI32 *, MemRefI64 *);

enum { kExtent = 8 };

static int64_t pack(int32_t real, int32_t imaginary) {
  return (int64_t)(((uint64_t)(uint32_t)imaginary << 32) | (uint32_t)real);
}

static void run(void (*kernel)(MemRefI32 *, MemRefI64 *), const int64_t *bins, int32_t *out) {
  int64_t input[kExtent];
  memcpy(input, bins, sizeof(input));
  MemRefI64 in = {input, input, 0, {kExtent}, {1}};
  MemRefI32 result;
  kernel(&result, &in);
  for (int i = 0; i < kExtent; ++i)
    out[i] = result.aligned[result.offset + i * result.strides[0]];
  free(result.allocated);
}

int main(void) {
  int64_t bins[kExtent];
  for (int i = 0; i < kExtent; ++i)
    bins[i] = pack(-1, 0);
  bins[1] = pack(-1, -1);
  bins[2] = pack(INT32_MIN, INT32_MIN);
  bins[3] = pack(0, 0);

  int32_t even[kExtent], floorArm[kExtent];
  run(_mlir_ciface_q31_magnitude_component_even, bins, even);
  run(_mlir_ciface_q31_magnitude_component_floor, bins, floorArm);

  int failures = 0;
  /* The named witness, not a sweep: re = -1 is exactly the tie the two rules
   * split. */
  if (even[0] != 0 || floorArm[0] != 2) {
    fprintf(stderr, "witness (-1, 0): even %d (want 0), floor %d (want 2)\n", even[0], floorArm[0]);
    ++failures;
  }
  /* The saturating corner must survive both rules. */
  if (even[2] != INT32_MAX || floorArm[2] != INT32_MAX) {
    fprintf(stderr, "corner (INT32_MIN, INT32_MIN): even %d floor %d, want %d\n", even[2],
            floorArm[2], INT32_MAX);
    ++failures;
  }
  if (even[3] != 0 || floorArm[3] != 0) {
    fprintf(stderr, "origin: even %d floor %d, want 0\n", even[3], floorArm[3]);
    ++failures;
  }
  int differ = 0;
  for (int i = 0; i < kExtent; ++i)
    if (even[i] != floorArm[i])
      differ = 1;
  if (!differ) {
    fprintf(stderr, "the two declared component roundings agreed on every bin\n");
    ++failures;
  }
  if (failures)
    return 1;
  printf("ox q31 magnitude component gate ok\n");
  return 0;
}
