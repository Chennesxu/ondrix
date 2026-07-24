def invalid_fir_filter_boundary(input: tensor[q15], coefficients: tensor[q15]) -> tensor[q15]:
  return fir_filter(input, coefficients, boundary=same, accumulator=exact[40,saturate], rounding=nearest_even, overflow=saturate)
