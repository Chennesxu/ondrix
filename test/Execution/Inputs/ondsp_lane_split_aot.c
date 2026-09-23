#include <stdint.h>
#include <stdio.h>

typedef struct {
  int16_t *allocated;
  int16_t *aligned;
  intptr_t offset;
  intptr_t size;
  intptr_t stride;
} MemRef1D;

int16_t _mlir_ciface_split_word_sum(MemRef1D *x);
int32_t _mlir_ciface_split_by_halves(MemRef1D *x);
int16_t _mlir_ciface_split_odd_tail(MemRef1D *x);
int16_t _mlir_ciface_split_declared(MemRef1D *x, MemRef1D *taps);

static const int16_t kWordSum[8] = {-1200, 2500, 6000, 9100, 9100, 6000, 2500, -1200};
static const int16_t kHalves[8] = {16383, 16381, 16383, 16379, 16383, 16381, 16383, 16379};
static const int16_t kOddTail[5] = {3000, -2000, 5000, 7000, 1000};

static uint32_t state = 0x2545F491u;
static uint32_t next(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

// The declared chain: an i40 saturating accumulator of full products, then
// floor((acc + half) / 2^shift) saturated to the destination.
static int64_t reference(const int16_t *x, const int16_t *taps, int count, int shift, int64_t half,
                         int64_t low, int64_t high) {
  const int64_t rail = ((int64_t)1 << 39) - 1;
  int64_t acc = 0;
  for (int i = 0; i < count; ++i) {
    acc += (int64_t)x[i] * taps[i];
    acc = acc > rail ? rail : (acc < -rail - 1 ? -rail - 1 : acc);
  }
  int64_t rounded = (acc + half) >> shift;
  return rounded > high ? high : (rounded < low ? low : rounded);
}

static int16_t storage[2][10] __attribute__((aligned(8)));
static int16_t tapStorage[2][10] __attribute__((aligned(8)));
// Runtime taps whose absolute sum stays below 2 << 15: the edge of the
// declared bound, and random tables scaled under it.
static int16_t declared[8];

static void drawDeclared(int trial) {
  static const int16_t edge[8] = {32767, 32767, 1, 0, 0, 0, 0, 0};
  int64_t sum = 0;
  for (int i = 0; i < 8; ++i) {
    declared[i] = trial % 5 == 0 ? edge[i] : (int16_t)((int32_t)next() % 8191);
    sum += declared[i] < 0 ? -declared[i] : declared[i];
  }
  if (sum >= 65536)
    for (int i = 0; i < 8; ++i)
      declared[i] /= 2;
}

static int16_t *place(int misaligned, const int16_t *values, int count) {
  int16_t *base = &storage[misaligned][misaligned];
  for (int i = 0; i < count; ++i)
    base[i] = values[i];
  return base;
}

int main(void) {
  long failures = 0, beyondWord = 0;
  for (int trial = 0; trial < 60000; ++trial) {
    int16_t x[8];
    int mode = trial % 4;
    for (int i = 0; i < 8; ++i) {
      int16_t random = (int16_t)next();
      // Rails and same-signed near-rail runs push the merged sum past a word.
      x[i] = mode == 0   ? random
             : mode == 1 ? (int16_t)(32767 - (next() & 0x3FF))
             : mode == 2 ? (int16_t)(-32768 + (next() & 0x3FF))
                         : ((next() & 1) ? 32767 : -32768);
    }
    int64_t sum = 0;
    for (int i = 0; i < 8; ++i)
      sum += (int64_t)x[i] * kHalves[i];
    beyondWord += sum > INT32_MAX || sum < INT32_MIN;
    for (int misaligned = 0; misaligned < 2; ++misaligned) {
      MemRef1D view = {0, 0, 0, 8, 1};
      view.allocated = view.aligned = place(misaligned, x, 8);
      int64_t got16 = _mlir_ciface_split_word_sum(&view);
      int64_t got32 = _mlir_ciface_split_by_halves(&view);
      view.size = 6;
      int64_t gotOdd = _mlir_ciface_split_odd_tail(&view);
      int64_t want16 = reference(x, kWordSum, 8, 15, 1 << 14, INT16_MIN, INT16_MAX);
      int64_t want32 = reference(x, kHalves, 8, 1, 0, INT32_MIN, INT32_MAX);
      int64_t wantOdd = reference(x, kOddTail, 5, 15, 1 << 14, INT16_MIN, INT16_MAX);
      for (int tapSkew = 0; tapSkew < 2; ++tapSkew) {
        drawDeclared(trial);
        int16_t *taps = &tapStorage[tapSkew][tapSkew];
        for (int i = 0; i < 8; ++i)
          taps[i] = declared[i];
        MemRef1D stream = {0, 0, 0, 8, 1};
        stream.allocated = stream.aligned = place(misaligned, x, 8);
        MemRef1D table = {taps, taps, 0, 8, 1};
        if (_mlir_ciface_split_declared(&stream, &table) !=
            reference(x, declared, 8, 15, 1 << 14, INT16_MIN, INT16_MAX))
          ++failures;
      }
      if (got16 != want16 || got32 != want32 || gotOdd != wantOdd) {
        if (failures < 8)
          printf("trial %d misaligned %d: %lld/%lld %lld/%lld %lld/%lld\n", trial, misaligned,
                 (long long)got16, (long long)want16, (long long)got32, (long long)want32,
                 (long long)gotOdd, (long long)wantOdd);
        ++failures;
      }
    }
  }
  // The halves merge is only exercised where the exact sum leaves a word.
  if (beyondWord < 1000) {
    printf("only %ld cases leave a word\n", beyondWord);
    return 1;
  }
  printf("lane split: %ld failures, %ld sums past a word\n", failures, beyondWord);
  return failures != 0;
}
