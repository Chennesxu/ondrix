#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MEMREF_ARGS(pointer, size) pointer, pointer, INT64_C(0), size, INT64_C(1)

extern void copy_right(int32_t *, int32_t *, int64_t, int64_t, int64_t);
extern void copy_left(int32_t *, int32_t *, int64_t, int64_t, int64_t);

static int check(const char *name, const int32_t *actual, const int32_t *expected) {
  if (memcmp(actual, expected, 5 * sizeof(int32_t)) == 0)
    return 0;
  fprintf(stderr, "%s overlap copy mismatch\n", name);
  return 1;
}

int main(void) {
  int32_t right[] = {1, 2, 3, 4, 5};
  const int32_t expected_right[] = {1, 1, 2, 3, 4};
  copy_right(MEMREF_ARGS(right, 5));

  int32_t left[] = {1, 2, 3, 4, 5};
  const int32_t expected_left[] = {2, 3, 4, 5, 5};
  copy_left(MEMREF_ARGS(left, 5));

  return check("right", right, expected_right) | check("left", left, expected_left);
}
