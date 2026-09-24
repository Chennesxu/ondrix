#ifndef ONDRIX_SUPPORT_VITERBIDECODING_H
#define ONDRIX_SUPPORT_VITERBIDECODING_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace ondrix {

// The frame every layer of Viterbi decoding admits, or the rule it breaks:
// K in [3, 7], two or three generators in [1, 2^K), and N * R symbols with N a
// positive multiple of 8 and N * R <= 16384, packed into N / 8 bytes.
inline std::optional<std::string> checkViterbiFrame(int64_t constraintLength,
                                                    llvm::ArrayRef<int64_t> polynomials,
                                                    int64_t symbols, int64_t bytes) {
  if (constraintLength < 3 || constraintLength > 7)
    return std::string("constraint_length must be in [3, 7]");
  if (polynomials.size() != 2 && polynomials.size() != 3)
    return std::string("requires two or three generator polynomials (rate 1/2 or 1/3)");
  for (int64_t polynomial : polynomials)
    if (polynomial < 1 || polynomial >= (int64_t{1} << constraintLength))
      return "generator polynomial " + std::to_string(polynomial) + " is not in [1, 2^" +
             std::to_string(constraintLength) + ")";
  int64_t rate = static_cast<int64_t>(polynomials.size());
  int64_t frame = symbols / rate;
  if (symbols % rate != 0 || frame == 0 || frame % 8 != 0 || symbols > 16384)
    return "requires N * R symbols with N a positive multiple of 8 and N * R <= 16384; got " +
           std::to_string(symbols) + " symbols at rate 1/" + std::to_string(rate);
  if (bytes != frame / 8)
    return "packs " + std::to_string(frame) + " decoded bits into " + std::to_string(frame / 8) +
           " bytes, not " + std::to_string(bytes);
  return std::nullopt;
}

// A finite-width realization of the path-metric recursion
// `ondrix.viterbi_decode` defines exactly, as `ondsp.viterbi_decode` carries
// it; the argument that it decodes exactly is in that operation's description.
struct ViterbiMetricRealization {
  int64_t metricBits;
  int64_t symbolBound;
  int64_t unreachableMetric;
  int64_t renormalizationPeriod;
};

// The realization every frame the ondrix operation admits decodes exactly with:
// i32 metrics over the whole i16 symbol range, where N * R <= 16384 keeps every
// reachable metric within 2^29, and a seed no reachable path can meet.
inline ViterbiMetricRealization getWideViterbiRealization() {
  return {32, 32768, -(int64_t{1} << 30) - 1, 0};
}

// Whether, over `stages` stages of symbols within the bound, every value the
// realization computes fits its width and the seed of the unreachable states
// loses every comparison it meets: then each decision is the exact one.
inline bool isExactViterbiRealization(int64_t constraintLength, int64_t rate, int64_t stages,
                                      const ViterbiMetricRealization &realization) {
  if (realization.metricBits < 2 || realization.metricBits > 32 || realization.symbolBound < 1 ||
      realization.symbolBound > 32768 || realization.renormalizationPeriod < 0)
    return false;
  const int64_t high = (int64_t{1} << (realization.metricBits - 1)) - 1;
  const int64_t low = -(int64_t{1} << (realization.metricBits - 1));
  const int64_t step = rate * realization.symbolBound;
  const int64_t reach = (constraintLength - 1) * step;
  if (step > high)
    return false;
  // Until stage K-1 an unreachable metric is the seed moved by at most `reach`.
  const int64_t seed = realization.unreachableMetric;
  if (seed + reach >= -reach || seed - reach < low)
    return false;
  const int64_t period = realization.renormalizationPeriod;
  if (period == 0 || period >= stages)
    return stages * step <= high;
  // From stage K-1 on every state is reachable and the spread is at most
  // 2 * reach, so subtracting state 0's metric leaves [-2 * reach, 2 * reach]
  // and a period adds at most one step each way per stage.
  return period >= constraintLength - 1 && 2 * reach + period * step <= high;
}

// The exact realization in `metricBits` bits with the fewest renormalizations,
// or none: the lowest seed that cannot wrap, and no renormalization when the
// frame's whole growth fits.
inline std::optional<ViterbiMetricRealization>
chooseViterbiRealization(int64_t constraintLength, int64_t rate, int64_t stages, int64_t metricBits,
                         int64_t symbolBound) {
  if (metricBits < 2 || metricBits > 32)
    return std::nullopt;
  const int64_t high = (int64_t{1} << (metricBits - 1)) - 1;
  const int64_t low = -(int64_t{1} << (metricBits - 1));
  const int64_t step = rate * symbolBound;
  ViterbiMetricRealization realization{metricBits, symbolBound, low + (constraintLength - 1) * step,
                                       0};
  if (step <= high && stages * step > high)
    realization.renormalizationPeriod = high / step - 2 * (constraintLength - 1);
  if (!isExactViterbiRealization(constraintLength, rate, stages, realization))
    return std::nullopt;
  return realization;
}

} // namespace ondrix

#endif // ONDRIX_SUPPORT_VITERBIDECODING_H
