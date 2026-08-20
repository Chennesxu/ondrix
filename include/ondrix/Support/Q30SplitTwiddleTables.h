#ifndef ONDRIX_SUPPORT_Q30SPLITTWIDDLETABLES_H
#define ONDRIX_SUPPORT_Q30SPLITTWIDDLETABLES_H

#include <cstdint>
#include <optional>

namespace ondrix {

// Frozen signed Q2.30 split-stage twiddles sin/cos(pi*k/32) for k = 0..16,
// quantized round-half-even. The words are evaluated offline at 50 decimal
// digits; the worst distance from a rounding tie over all thirty-four is
// 0.048 LSB and a binary64 llround chain agrees on every one of the
// seventeen pairs, so these are the correctly rounded integers. Smaller
// extents stride the same quadrant by 32/N, so nothing else is tabulated.

// Largest complex extent the frozen table covers. A larger extent must fail
// closed rather than extrapolate.
inline constexpr int64_t kMaxQ30SplitTwiddleExtent = 32;

struct Q30SplitTwiddle {
  int64_t sine;
  int64_t cosine;
};

inline constexpr int64_t kQ30SplitTwiddles[kMaxQ30SplitTwiddleExtent / 2 + 1][2] = {
    {0, 1073741824},         {105245103, 1068571464}, {209476638, 1053110176},
    {311690799, 1027506862}, {410903207, 992008094},  {506158392, 946955747},
    {596538995, 892783698},  {681174602, 830013654},  {759250125, 759250125},
    {830013654, 681174602},  {892783698, 596538995},  {946955747, 506158392},
    {992008094, 410903207},  {1027506862, 311690799}, {1053110176, 209476638},
    {1068571464, 105245103}, {1073741824, 0},
};

// The split pair at angle pi*index/extent, reached by striding the frozen
// quadrant. Returns std::nullopt outside it, including for an extent that
// does not divide the tabulated one, so every consumer fails closed.
inline std::optional<Q30SplitTwiddle> getQ30SplitTwiddle(int64_t extent, int64_t index) {
  if (extent < 2 || extent > kMaxQ30SplitTwiddleExtent || kMaxQ30SplitTwiddleExtent % extent != 0 ||
      index < 0 || index > extent / 2)
    return std::nullopt;
  int64_t slot = index * (kMaxQ30SplitTwiddleExtent / extent);
  return Q30SplitTwiddle{kQ30SplitTwiddles[slot][0], kQ30SplitTwiddles[slot][1]};
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_Q30SPLITTWIDDLETABLES_H
