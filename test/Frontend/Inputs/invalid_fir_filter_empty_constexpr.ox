def invalid_fir_filter_empty_constexpr(
    input: tensor[q15,9223372036854775807],
    coefficients: constexpr[q15] = []) -> tensor[q15,1]:
  return fir_filter(input, coefficients, boundary=full, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
