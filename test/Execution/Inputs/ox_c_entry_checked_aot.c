#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

// The descriptor entries the checked plain entries are held against.
extern int16_t q15_dot(int16_t *lhs_allocated, int16_t *lhs_aligned, int64_t lhs_offset,
                       int64_t lhs_size, int64_t lhs_stride, int16_t *rhs_allocated,
                       int16_t *rhs_aligned, int64_t rhs_offset, int64_t rhs_size,
                       int64_t rhs_stride);
typedef struct {
  int16_t *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRefI16;
extern void _mlir_ciface_q15_multi_use_binding(MemRefI16 *, MemRefI16 *, MemRefI16 *);

enum { kLength = 32 };

// Runs `call` in a child; true when it aborted after printing `message`.
static int refuses(void (*call)(int16_t *), int16_t *page, const char *message) {
  int pipefd[2];
  if (pipe(pipefd) != 0)
    return 0;
  pid_t child = fork();
  if (child == 0) {
    dup2(pipefd[1], STDOUT_FILENO);
    close(pipefd[0]);
    setvbuf(stdout, NULL, _IONBF, 0);
    call(page);
    _exit(0);
  }
  close(pipefd[1]);
  // Read to end of file: closing the pipe early would kill the child's second
  // unbuffered write with SIGPIPE before it reaches abort.
  char output[256] = {0};
  size_t count = 0;
  for (ssize_t got = 1; got > 0 && count < sizeof(output) - 1; count += (size_t)got)
    got = read(pipefd[0], output + count, sizeof(output) - 1 - count);
  close(pipefd[0]);
  int status = 0;
  if (child < 0 || waitpid(child, &status, 0) != child)
    return 0;
  if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT) {
    fprintf(stderr, "%s: not refused\n", message);
    return 0;
  }
  if (count == 0 || !strstr(output, message)) {
    fprintf(stderr, "%s: refused with '%s'\n", message, output);
    return 0;
  }
  return 1;
}

static void dotSameBuffer(int16_t *page) { ondrix_q15_dot(page, page, kLength); }
static void dotPartialOverlap(int16_t *page) { ondrix_q15_dot(page, page + 4, kLength); }
static void dotNullWithElements(int16_t *page) { ondrix_q15_dot(NULL, page, kLength); }
static void chainOutputOverInput(int16_t *page) {
  ondrix_q15_multi_use_binding(page, page + kLength, page + 8);
}

static int16_t sample(uint32_t *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return (int16_t)((int32_t)(*state % 65536u) - 32768);
}

int main(void) {
  // A page the parent can read after a child aborted: the refusal must
  // happen before any store, so the sentinel pattern survives untouched.
  int16_t *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (page == MAP_FAILED)
    return 1;
  uint32_t state = 0x2F6E19A3u;
  for (int i = 0; i < 3 * kLength; ++i)
    page[i] = sample(&state);
  int16_t sentinel[3 * kLength];
  memcpy(sentinel, page, sizeof(sentinel));

  if (!refuses(dotSameBuffer, page, "ondrix_q15_dot: two buffers overlap") ||
      !refuses(dotPartialOverlap, page, "ondrix_q15_dot: two buffers overlap") ||
      !refuses(dotNullWithElements, page, "ondrix_q15_dot: a null buffer has elements") ||
      !refuses(chainOutputOverInput, page, "ondrix_q15_multi_use_binding: two buffers overlap"))
    return 1;
  if (memcmp(sentinel, page, sizeof(sentinel)) != 0) {
    fprintf(stderr, "a refused call wrote into its buffers\n");
    return 1;
  }

  // Valid calls are unchanged: adjacent buffers are disjoint, empty ranges
  // may share an address or be null.
  int16_t *lhs = page, *rhs = page + kLength, *output = page + 2 * kLength;
  int16_t direct = q15_dot(lhs, lhs, 0, kLength, 1, rhs, rhs, 0, kLength, 1);
  if (ondrix_q15_dot(lhs, rhs, kLength) != direct || ondrix_q15_dot(lhs, lhs, 0) != 0 ||
      ondrix_q15_dot(NULL, NULL, 0) != 0) {
    fprintf(stderr, "a valid dot call was mishandled\n");
    return 1;
  }
  int16_t expected[kLength];
  MemRefI16 lhsRef = {lhs, lhs, 0, {kLength}, {1}};
  MemRefI16 rhsRef = {rhs, rhs, 0, {kLength}, {1}};
  MemRefI16 expectedRef = {expected, expected, 0, {kLength}, {1}};
  _mlir_ciface_q15_multi_use_binding(&lhsRef, &rhsRef, &expectedRef);
  ondrix_q15_multi_use_binding(lhs, rhs, output);
  if (memcmp(expected, output, sizeof(expected)) != 0) {
    fprintf(stderr, "a valid chain call was mishandled\n");
    return 1;
  }
  return 0;
}
