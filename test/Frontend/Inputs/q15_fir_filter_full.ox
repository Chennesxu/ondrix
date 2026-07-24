def q15_fir_filter_full(input: tensor[q15,6], coefficients: tensor[q15,3]) -> tensor[q15,8]:
  return fir_filter(input, coefficients, boundary=full, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
