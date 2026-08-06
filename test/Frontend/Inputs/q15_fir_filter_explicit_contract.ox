def q15_fir_filter_explicit_contract(
    input: tensor[q15,64], coefficients: tensor[q15,8]) -> tensor[q15,57]:
  return fir_filter(input, coefficients, boundary=valid,
                    accumulator=exact[40,saturate], rounding=nearest_ties_positive, overflow=saturate)
