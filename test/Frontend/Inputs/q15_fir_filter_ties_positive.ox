def q15_fir_filter_ties_positive(input: tensor[q15], coefficients: tensor[q15]) -> tensor[q15]:
  return fir_filter(input, coefficients, boundary=valid, accumulator=exact[40,saturate], rounding=nearest_ties_positive, overflow=saturate)
