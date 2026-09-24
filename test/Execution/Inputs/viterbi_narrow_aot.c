#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  void *allocated;
  void *aligned;
  intptr_t offset;
  intptr_t size;
  intptr_t stride;
} MemRef1D;

typedef void (*Decoder)(MemRef1D *, MemRef1D *);

void _mlir_ciface_k7_short(MemRef1D *, MemRef1D *);
void _mlir_ciface_k7_long(MemRef1D *, MemRef1D *);
void _mlir_ciface_k7_rail(MemRef1D *, MemRef1D *);
#ifndef TRELLIS_ONLY
void _mlir_ciface_k5_r3(MemRef1D *, MemRef1D *);
void _mlir_ciface_k3_r2(MemRef1D *, MemRef1D *);
#endif

typedef struct {
  const char *name;
  int constraint;
  int rate;
  int polynomials[3];
  int frame;
  int bound;
  Decoder decode;
} Code;

enum { MAX_FRAME = 512, MAX_STATES = 64 };

static int parity(unsigned value) {
  value ^= value >> 16;
  value ^= value >> 8;
  value ^= value >> 4;
  value ^= value >> 2;
  value ^= value >> 1;
  return value & 1;
}

static int coded_bit(const Code *code, int r, int input, int state) {
  unsigned reg = ((unsigned)input << (code->constraint - 1)) | (unsigned)state;
  return parity((unsigned)code->polynomials[r] & reg);
}

/* The contract read literally, as test/Execution/Inputs/viterbi_aot.c reads
 * it: exact 64-bit metrics and reachability instead of a finite -infinity. */
static void reference(const Code *code, const int16_t *symbols, uint8_t *bits) {
  static unsigned char survivor[MAX_FRAME][MAX_STATES];
  int states = 1 << (code->constraint - 1);
  int64_t metric[MAX_STATES], next[MAX_STATES];
  int reachable[MAX_STATES], reached[MAX_STATES];
  for (int s = 0; s < states; ++s) {
    metric[s] = 0;
    reachable[s] = s == 0;
  }
  for (int n = 0; n < code->frame; ++n) {
    for (int t = 0; t < states; ++t) {
      int input = t >> (code->constraint - 2);
      int pred[2] = {2 * (t & (states / 2 - 1)), 2 * (t & (states / 2 - 1)) + 1};
      int64_t candidate[2];
      for (int k = 0; k < 2; ++k) {
        int64_t gain = 0;
        for (int r = 0; r < code->rate; ++r) {
          int16_t y = symbols[n * code->rate + r];
          gain += coded_bit(code, r, input, pred[k]) ? -y : y;
        }
        candidate[k] = metric[pred[k]] + gain;
      }
      int pick;
      if (reachable[pred[0]] && reachable[pred[1]])
        pick = candidate[0] > candidate[1] ? 0 : 1;
      else
        pick = reachable[pred[0]] ? 0 : 1;
      next[t] = candidate[pick];
      survivor[n][t] = (unsigned char)pred[pick];
      reached[t] = reachable[pred[0]] || reachable[pred[1]];
    }
    memcpy(metric, next, sizeof metric);
    memcpy(reachable, reached, sizeof reachable);
  }
  memset(bits, 0, (size_t)code->frame / 8);
  int state = 0;
  for (int n = code->frame - 1; n >= 0; --n) {
    if (state >> (code->constraint - 2))
      bits[n / 8] |= (uint8_t)(0x80u >> (n % 8));
    state = survivor[n][state];
  }
}

static uint32_t lcg = 0x3C6EF372u;
static uint32_t next_random(void) {
  lcg = lcg * 1664525u + 1013904223u;
  return lcg >> 8;
}

static int16_t clamp(int value, int bound) {
  return (int16_t)(value > bound ? bound : value < -bound ? -bound : value);
}

static int failures = 0;

static void check(const Code *code, const char *what, int16_t *symbols) {
  static uint8_t want[MAX_FRAME / 8];
  static int8_t got[MAX_FRAME / 8];
  MemRef1D in = {symbols, symbols, 0, code->frame * code->rate, 1};
  MemRef1D out = {got, got, 0, code->frame / 8, 1};
  reference(code, symbols, want);
  code->decode(&in, &out);
  if (memcmp(want, got, (size_t)code->frame / 8) != 0) {
    printf("%s %s: decoder differs from the reference\n", code->name, what);
    ++failures;
  }
}

/* Rail frames are noiseless codewords at the declared bound: the best path
 * gains the full R * B every stage, so only the renormalization keeps it in
 * range. Noisy and uniform frames spread the metrics; zeros tie everything. */
static void run_code(const Code *code) {
  static int16_t symbols[MAX_FRAME * 3];
  int count = code->frame * code->rate;
  for (int frame = 0; frame < 12; ++frame) {
    int state = 0;
    for (int n = 0; n < code->frame; ++n) {
      int input = n < code->frame - (code->constraint - 1) ? (int)(next_random() & 1) : 0;
      for (int r = 0; r < code->rate; ++r) {
        int sign = coded_bit(code, r, input, state) ? -1 : 1;
        int noise =
            frame % 3 == 0 ? 0 : (int)(next_random() % (uint32_t)code->bound) - code->bound / 2;
        symbols[n * code->rate + r] = clamp(sign * code->bound + noise, code->bound);
      }
      state = (state >> 1) | (input << (code->constraint - 2));
    }
    check(code, frame % 3 == 0 ? "rail frame" : "noisy frame", symbols);
  }
  for (int i = 0; i < count; ++i)
    symbols[i] =
        clamp((int)(next_random() % (uint32_t)(2 * code->bound + 1)) - code->bound, code->bound);
  check(code, "uniform frame", symbols);
  memset(symbols, 0, sizeof symbols);
  check(code, "all-tie frame", symbols);
}

int main(void) {
  const Code codes[] = {
      {"k7_short", 7, 2, {0171, 0133, 0}, 64, 127, _mlir_ciface_k7_short},
      {"k7_long", 7, 2, {0171, 0133, 0}, 512, 127, _mlir_ciface_k7_long},
      {"k7_rail", 7, 2, {0171, 0133, 0}, 512, 900, _mlir_ciface_k7_rail},
#ifndef TRELLIS_ONLY
      {"k5_r3", 5, 3, {025, 033, 037}, 64, 200, _mlir_ciface_k5_r3},
      {"k3_r2", 3, 2, {07, 05, 0}, 256, 2000, _mlir_ciface_k3_r2},
#endif
  };
  for (unsigned i = 0; i < sizeof codes / sizeof codes[0]; ++i)
    run_code(&codes[i]);
  if (failures) {
    printf("FAIL: %d\n", failures);
    return 1;
  }
  printf("viterbi narrowed metrics: PASS\n");
  return 0;
}
