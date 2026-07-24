def q15_fir_filter_valid(input: tensor[q15], coefficients: tensor[q15]) -> tensor[q15]:
  return fir_filter(input, coefficients, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
