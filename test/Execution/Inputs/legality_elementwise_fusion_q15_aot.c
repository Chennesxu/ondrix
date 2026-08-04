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

typedef void (*Kernel)(MemRefI16 *, MemRefI16 *);

/* Certified pairs, compiled with and without the fusion pass. */
extern void _mlir_ciface_negate_wrap_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fused_negate_wrap_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_directed_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fused_shift_directed_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_offset_wrap_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fused_offset_wrap_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_abs_of_negate_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_fused_abs_of_negate_pair(MemRefI16 *, MemRefI16 *);

/* Refused pairs, compiled together with the single operation the rewrite
 * would have produced. Each must DIFFER, or the refusal was unnecessary. */
extern void _mlir_ciface_negate_saturate_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_negate_saturate_identity(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_nearest_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_shift_nearest_single(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_offset_saturate_pair(MemRefI16 *, MemRefI16 *);
extern void _mlir_ciface_offset_saturate_identity(MemRefI16 *, MemRefI16 *);

enum { kBlock = 4096, kBlocks = 16 };

static int failures;

/* A chain that collapses to the identity returns the caller's own buffer
 * through the bufferized descriptor, so ownership is decided by comparing
 * the allocation against the input. This is the test wrapper's convention,
 * not a source-level ownership model. */
static void release(MemRefI16 *ref, const int16_t *input) {
  if (ref->allocated != input)
    free(ref->allocated);
}

/* The certificate decides the rewrite from a model of the contract. This
 * gate checks the other half: that the emitted objects agree over the same
 * whole 2^16 domain the certificate enumerated. A certificate that models
 * arithmetic the lowering does not implement would pass compile time and
 * fail here. */
static void requireIdentical(Kernel plain, Kernel fused, const char *label) {
  int16_t input[kBlock];
  for (int64_t block = 0; block < kBlocks; ++block) {
    for (int64_t i = 0; i < kBlock; ++i)
      input[i] = (int16_t)(int32_t)(block * kBlock + i - 32768);
    MemRefI16 inputRef = {input, input, 0, {kBlock}, {1}};
    MemRefI16 a;
    MemRefI16 b;
    plain(&a, &inputRef);
    fused(&b, &inputRef);
    for (int64_t i = 0; i < kBlock; ++i) {
      int16_t left = a.aligned[a.offset + i];
      int16_t right = b.aligned[b.offset + i];
      if (left != right && failures++ < 8)
        fprintf(stderr, "%s(%d): unfused %d, fused %d\n", label, input[i], left, right);
    }
    release(&a, input);
    release(&b, input);
  }
}

/* A refusal is only evidence if the rewrite it refused would have been
 * wrong. The witness input is reported so the divergence is a fact in the
 * log rather than a claim in a comment. */
static void requireDifferent(Kernel pair, Kernel single, const char *label) {
  int16_t input[kBlock];
  int found = 0;
  int16_t witness = 0;
  int16_t got = 0;
  int16_t would = 0;
  for (int64_t block = 0; block < kBlocks && !found; ++block) {
    for (int64_t i = 0; i < kBlock; ++i)
      input[i] = (int16_t)(int32_t)(block * kBlock + i - 32768);
    MemRefI16 inputRef = {input, input, 0, {kBlock}, {1}};
    MemRefI16 a;
    MemRefI16 b;
    pair(&a, &inputRef);
    single(&b, &inputRef);
    for (int64_t i = 0; i < kBlock && !found; ++i)
      if (a.aligned[a.offset + i] != b.aligned[b.offset + i]) {
        found = 1;
        witness = input[i];
        got = a.aligned[a.offset + i];
        would = b.aligned[b.offset + i];
      }
    release(&a, input);
    release(&b, input);
  }
  if (!found) {
    fprintf(stderr, "%s: the refused rewrite agrees everywhere\n", label);
    ++failures;
    return;
  }
  printf("%s: refused correctly, %d gives %d but the rewrite gives %d\n", label, witness, got,
         would);
}

int main(void) {
  requireIdentical(_mlir_ciface_negate_wrap_pair, _mlir_ciface_fused_negate_wrap_pair,
                   "negate_wrap");
  requireIdentical(_mlir_ciface_shift_directed_pair, _mlir_ciface_fused_shift_directed_pair,
                   "shift_directed");
  requireIdentical(_mlir_ciface_offset_wrap_pair, _mlir_ciface_fused_offset_wrap_pair,
                   "offset_wrap");
  requireIdentical(_mlir_ciface_abs_of_negate_pair, _mlir_ciface_fused_abs_of_negate_pair,
                   "abs_of_negate");

  requireDifferent(_mlir_ciface_negate_saturate_pair, _mlir_ciface_negate_saturate_identity,
                   "negate_saturate");
  requireDifferent(_mlir_ciface_shift_nearest_pair, _mlir_ciface_shift_nearest_single,
                   "shift_nearest");
  requireDifferent(_mlir_ciface_offset_saturate_pair, _mlir_ciface_offset_saturate_identity,
                   "offset_saturate");
  return failures != 0;
}
