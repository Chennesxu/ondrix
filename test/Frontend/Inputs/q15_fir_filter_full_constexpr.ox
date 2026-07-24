def q15_fir_filter_full_constexpr(
    input: tensor[q15,8],
    coefficients: constexpr[q15] = [1024, -512, 256, 128]) -> tensor[q15,11]:
  return fir_filter(input, coefficients, boundary=full, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
