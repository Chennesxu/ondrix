#include <stdint.h>
#include <stdio.h>
#include <string.h>

void ondrix_viterbi_k7_r2(const int16_t *symbols, int8_t *bits);
void ondrix_viterbi_k5_r3(const int16_t *symbols, int8_t *bits);
void ondrix_viterbi_k3_r2(const int16_t *symbols, int8_t *bits);
void ondrix_viterbi_k7_bound(const int16_t *symbols, int8_t *bits);

typedef void (*Decoder)(const int16_t *, int8_t *);

typedef struct {
  const char *name;
  int constraint;
  int rate;
  int polynomials[3];
  int frame;
  int frames;
  Decoder decode;
} Code;

enum { MAX_FRAME = 8192, MAX_STATES = 64 };

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

static void encode(const Code *code, const uint8_t *message, int amplitude, int16_t *symbols) {
  int state = 0;
  for (int n = 0; n < code->frame; ++n) {
    for (int r = 0; r < code->rate; ++r)
      symbols[n * code->rate + r] =
          (int16_t)(coded_bit(code, r, message[n], state) ? -amplitude : amplitude);
    state = (state >> 1) | (message[n] << (code->constraint - 2));
  }
}

/* The contract read literally: reachability instead of a finite -infinity,
 * exact 64-bit metrics, one survivor predecessor per state and stage. */
static void reference(const Code *code, const int16_t *symbols, int tie_odd, uint8_t *bits) {
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
        pick = candidate[0] > candidate[1]   ? 0
               : candidate[1] > candidate[0] ? 1
                                             : (tie_odd ? 1 : 0);
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
    int input = state >> (code->constraint - 2);
    if (input)
      bits[n / 8] |= (uint8_t)(0x80u >> (n % 8));
    state = survivor[n][state];
  }
}

static uint32_t lcg = 0x2545F491u;
static uint32_t next_random(void) {
  lcg = lcg * 1664525u + 1013904223u;
  return lcg >> 8;
}

static int16_t clamp16(int value) {
  return (int16_t)(value > 32767 ? 32767 : value < -32768 ? -32768 : value);
}

static int failures = 0;

static void expect_equal(const Code *code, const char *what, const uint8_t *want,
                         const int8_t *got) {
  if (memcmp(want, got, (size_t)code->frame / 8) != 0) {
    printf("%s %s: decoder differs from the reference\n", code->name, what);
    ++failures;
  }
}

static void run_code(const Code *code) {
  static uint8_t message[MAX_FRAME], packed[MAX_FRAME / 8], want[MAX_FRAME / 8];
  static int16_t symbols[MAX_FRAME * 3];
  static int8_t got[MAX_FRAME / 8];
  int bytes = code->frame / 8;
  for (int frame = 0; frame < code->frames; ++frame) {
    memset(packed, 0, sizeof packed);
    for (int n = 0; n < code->frame; ++n) {
      message[n] = n < code->frame - (code->constraint - 1) ? (uint8_t)(next_random() & 1) : 0;
      if (message[n])
        packed[n / 8] |= (uint8_t)(0x80u >> (n % 8));
    }
    /* Noiseless frames must decode to the message itself. */
    encode(code, message, 64, symbols);
    code->decode(symbols, got);
    if (memcmp(packed, got, (size_t)bytes) != 0) {
      printf("%s frame %d: noiseless frame not recovered\n", code->name, frame);
      ++failures;
    }
    /* Noisy and full-scale frames must match the reference bit for bit. */
    int amplitude = frame % 3 == 0 ? 32767 : 900;
    int noise = frame % 3 == 0 ? 65535 : frame % 3 == 1 ? 1200 : 2600;
    encode(code, message, amplitude, symbols);
    for (int i = 0; i < code->frame * code->rate; ++i)
      symbols[i] = clamp16(symbols[i] + (int)(next_random() % (uint32_t)(noise + 1)) - noise / 2);
    reference(code, symbols, 1, want);
    code->decode(symbols, got);
    expect_equal(code, "noisy frame", want, got);
  }
  /* Every comparison ties: the odd-predecessor rule alone decides, and the
   * even one would decode something else. */
  memset(symbols, 0, sizeof symbols);
  static uint8_t even[MAX_FRAME / 8];
  reference(code, symbols, 1, want);
  reference(code, symbols, 0, even);
  if (memcmp(want, even, (size_t)bytes) == 0) {
    printf("%s: the tie rule is not observable on the all-tie frame\n", code->name);
    ++failures;
  }
  code->decode(symbols, got);
  expect_equal(code, "all-tie frame", want, got);
}

int main(void) {
  const Code codes[] = {
      {"k7_r2", 7, 2, {0171, 0133, 0}, 256, 24, ondrix_viterbi_k7_r2},
      {"k5_r3", 5, 3, {025, 033, 037}, 64, 24, ondrix_viterbi_k5_r3},
      {"k3_r2", 3, 2, {07, 05, 0}, 16, 24, ondrix_viterbi_k3_r2},
      {"k7_bound", 7, 2, {0171, 0133, 0}, 8192, 3, ondrix_viterbi_k7_bound},
  };
  for (unsigned i = 0; i < sizeof codes / sizeof codes[0]; ++i)
    run_code(&codes[i]);
  if (failures) {
    printf("FAIL: %d\n", failures);
    return 1;
  }
  printf("viterbi_decode: PASS\n");
  return 0;
}
